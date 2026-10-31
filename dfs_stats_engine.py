"""Stat-driven NFL DFS simulator.

Football statistics are simulated first; DraftKings points are scored from those
outcomes.  Offensive players and the opposing DST share the same game scenario.
This is intentionally a transparent game-level model, not a play-by-play engine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from injury_opportunity import redistribute
from ai2_nfl_snap_roles import apply_snap_roles_to_rates

TEAM_ALIASES={"JAC":"JAX","WAS":"WSH","LA":"LAR"}

def _team(x):
    s=str(x).upper().strip(); return TEAM_ALIASES.get(s,s)

def _clean_name(x):
    """Canonical player identity for cross-source matching."""
    s=" ".join(str(x).lower().replace(".","").replace("'","").replace("-"," ").split())
    parts=s.split()
    suffixes={"jr","sr","ii","iii","iv","v"}
    while parts and parts[-1] in suffixes:
        parts.pop()
    return " ".join(parts)

def _opp(game, team):
    g=str(game).split()[0]
    if "@" not in g: return ""
    a,h=g.split("@",1); t=_team(team)
    return _team(h if _team(a)==t else a)

def _pa_points(points):
    p=np.asarray(points)
    return np.select([p==0,p<=6,p<=13,p<=20,p<=27,p<=34],[10,7,4,1,0,-1],default=-4).astype(float)

def _safe(v, default):
    try:
        z=float(v); return z if np.isfinite(z) else float(default)
    except Exception: return float(default)

def _multinomial_rows(rng, totals, probs):
    return np.asarray([rng.multinomial(int(max(0,n)), probs) for n in totals],dtype=np.int16)

def _team_scoring_budget(historical_pass_tds, historical_rush_tds, expected_field_goals=1.65):
    """Historical-rate-only scoring prior; no sportsbook inputs.

    This is a heuristic, not a calibrated or independently validated model.
    """
    passing = float(np.clip(_safe(historical_pass_tds, 1.45), 0.0, 5.0))
    rushing = float(np.clip(_safe(historical_rush_tds, 0.85), 0.0, 4.0))
    expected_tds = float(np.clip(passing + rushing, 0.35, 4.8))
    field_goals = float(np.clip(_safe(expected_field_goals, 1.65), 0.0, 3.5))
    return expected_tds, field_goals

def _td_allocation_with_capacity(rng, n_tds, weights, capacities):
    """Allocate TDs among players with actual eligible scoring events.

    One catch/carry cannot generate multiple touchdowns.  `weights` may
    incorporate role/goal-line evidence but never overrides event capacity.
    """
    remaining=np.asarray(capacities,dtype=np.int64).copy()
    weights=np.asarray(weights,dtype=float)
    result=np.zeros(len(remaining),dtype=np.int16)
    if n_tds>int(remaining.sum()):
        raise ValueError('More touchdowns than eligible receptions/carries')
    for _ in range(int(n_tds)):
        eligible=remaining>0
        probs=np.where(eligible,np.maximum(weights,0.),0.)
        if probs.sum()<=0:
            probs=eligible.astype(float)
        probs=probs/probs.sum()
        idx=int(rng.choice(len(probs),p=probs))
        result[idx]+=1
        remaining[idx]-=1
    return result

# Name variants are only used after exact current-team matching fails.
_FIRST_NAMES = {"matt": "matthew", "mike": "michael", "josh": "joshua", "will": "william", "rob": "robert"}

def _identity_name(name):
    parts = _clean_name(name).split()
    if parts:
        parts[0] = _FIRST_NAMES.get(parts[0], parts[0])
    return " ".join(parts)

def _volume(row, pos):
    fields = ('pass_attempts_pg', 'carries_pg') if pos == 'QB' else ('targets_pg', 'carries_pg')
    return sum(max(0., _safe(row.get(f), 0.)) for f in fields)

def _role_proxy(player):
    """Conservative explicit prior for no-history depth players, not a measured rate.

    The ceiling is deliberately small: without independent role evidence we must
    not turn a high DK projection into a large invented historical workload.
    """
    pos = str(player['Position']).upper()
    proj = max(0., _safe(player.get('proj'), 0.))
    base = dict(player=str(player['Name']), team=_team(player['TeamAbbrev']),
                position=pos, games_played=0, game='', source='ROLE_PROXY_NO_HISTORY',
                latest_observed_season=np.nan, latest_observed_week=np.nan)
    for col in ('pass_attempts_pg','completions_pg','pass_yards_pg','passing_tds_pg',
                'interceptions_pg','sacks_suffered_pg','carries_pg','rushing_yards_pg',
                'rushing_tds_pg','targets_pg','receptions_pg','receiving_yards_pg',
                'receiving_tds_pg','fumbles_lost_pg'):
        base[col] = 0.
    if pos == 'QB':
        base.update(pass_attempts_pg=18., completions_pg=11., pass_yards_pg=130.,
                    passing_tds_pg=.65, interceptions_pg=.6, sacks_suffered_pg=2.,
                    carries_pg=1.5, rushing_yards_pg=6.)
    elif pos == 'RB':
        base.update(carries_pg=min(3., max(.35, proj / 2.8)),
                    rushing_yards_pg=min(12., max(1., proj * 1.5)),
                    targets_pg=min(1.5, max(.2, proj / 4.5)),
                    receptions_pg=min(1., max(.1, proj / 6.)),
                    receiving_yards_pg=min(8., max(1., proj)))
    elif pos in ('WR', 'TE'):
        base.update(targets_pg=min(2., max(.3, proj / 3.5)),
                    receptions_pg=min(1.3, max(.2, proj / 5.)),
                    receiving_yards_pg=min(15., max(2., proj * 1.5)))
    return base

def _resolve_baselines(pool, rates):
    """Match exact-team first; uniquely identified transfers second; proxy last.

    Reject ambiguous names rather than linking the wrong athlete. Returns a
    per-player provenance report, exposed in stat_draws metadata.
    """
    r = rates.copy()
    r['team'] = r['team'].map(_team)
    r['_name'] = r['player'].map(_clean_name)
    r['_identity'] = r['player'].map(_identity_name)
    report = []
    additions = []
    transferred = set()
    replaced = set()
    eligible = pool[pool['Position'].astype(str).isin(['QB','RB','WR','TE'])]
    for _, player in eligible.iterrows():
        name = _clean_name(player['Name']); team = _team(player['TeamAbbrev'])
        pos = str(player['Position']).upper()
        current = r[(r['_name'] == name) & (r['team'] == team) & (r['position'].astype(str).str.upper() == pos)]
        if not current.empty and _volume(current.iloc[-1], pos) >= .05:
            continue
        if not current.empty:
            replaced.add((name, team))
        # Prefer uniquely identified historical player of the same position.
        key = _identity_name(player['Name'])
        candidates = r[(r['_identity'] == key) & (r['position'].astype(str).str.upper() == pos)]
        identities = candidates['_name'].dropna().unique()
        if len(identities) > 1:
            raise ValueError(f'Ambiguous historical identity for {player["Name"]} ({team}); manual player ID needed.')
        candidates = candidates[candidates.apply(lambda z: _volume(z, pos) >= .05, axis=1)]
        if not candidates.empty:
            src = candidates.sort_values('games_played', ascending=False).iloc[0].copy()
            new = src.to_dict(); old_team = str(src['team'])
            new.update(player=str(player['Name']), team=team, position=pos,
                       _name=name, _identity=key, source='HISTORICAL_TRANSFER_OR_ALIAS')
            additions.append(new)
            if old_team != team:
                transferred.add((str(src['_name']), old_team))
            report.append(dict(player=str(player['Name']), team=team, method='historical_transfer_or_alias', previous_team=old_team))
        else:
            new = _role_proxy(player)
            new.update(_name=name, _identity=key)
            additions.append(new)
            report.append(dict(player=str(player['Name']), team=team, method='conservative_role_proxy', previous_team=''))
    if transferred or replaced:
        removed = transferred | replaced
        r = r[~r.apply(lambda z: (str(z['_name']), str(z['team'])) in removed, axis=1)].copy()
    if additions:
        r = pd.concat([r, pd.DataFrame(additions)], ignore_index=True)
    return r, report

def simulate_stat_driven_dfs(pool, rates, n_sims=10000, seed=20261009, out_players=None, allow_role_proxies=True,
                             team_environment=None, snap_profiles=None):
    """Return (sim_players, dk_points, stat_draws).

    rates is the pregame player baseline produced by props_data.make_rates().
    Sacks are explicit shared events: they reduce QB dropbacks available for pass
    attempts and score +1 for the opposing DST. Turnovers, TDs and points allowed
    are also shared between offense and DST.
    """
    out_players=out_players or []
    team_environment=team_environment or {}
    snap_profiles=snap_profiles or {}
    out_keys={(_clean_name(item['player']),_team(item['team'])) for item in out_players}
    x=pool[pool['optimizer_eligible']==True].copy().reset_index(drop=True)
    if out_keys:
        x=x[[(_clean_name(row['Name']),_team(row['TeamAbbrev'])) not in out_keys
             for _,row in x.iterrows()]].copy().reset_index(drop=True)
    r, baseline_report = _resolve_baselines(x, rates)
    # Unified verified-data path fails closed; legacy callers retain their
    # existing fallback behavior until separately migrated and tested.
    if not allow_role_proxies:
        proxy_players = [f"{entry['player']} ({entry['team']})"
                         for entry in baseline_report
                         if entry.get('method') == 'conservative_role_proxy']
        if proxy_players:
            raise ValueError('Unobserved role proxies are forbidden: '
                             + ', '.join(proxy_players[:20]))
    x['TeamAbbrev']=x['TeamAbbrev'].map(_team); x['_name']=x['Name'].map(_clean_name)
    statcols=[c for c in r.columns if c not in {'player','team','position','game','source','latest_observed_season','latest_observed_week'}]
    # Never generate zero-volume player stats and assign arbitrary TDs.
    # Such rows previously yielded 0/6/12 DK-point distributions.
    eligible=x[x['Position'].astype(str).isin(['QB','RB','WR','TE'])]
    baseline_keys=set(zip(r['_name'].astype(str),r['team'].astype(str)))
    unmatched=eligible[[ (str(row['_name']),str(row['TeamAbbrev'])) not in baseline_keys
                         for _,row in eligible.iterrows() ]]
    if not unmatched.empty:
        details=', '.join(f"{row['Name']} ({row['TeamAbbrev']})" for _,row in unmatched.head(20).iterrows())
        raise ValueError(
            f'Stat baseline identity mismatch for {len(unmatched)} eligible offensive players: '
            +details+'. Canonical name + current-team matching failed after suffix normalization; '
            'simulation aborted rather than producing TD-only fantasy scores.'
        )
    # Even matched names are unusable when their historical opportunity columns
    # are absent/empty. Reject those rows before the optimizer can trust them.
    bad_rates=[]
    for _,row in eligible.iterrows():
        matched=r[(r['_name']==str(row['_name'])) & (r['team']==str(row['TeamAbbrev']))]
        if matched.empty: continue
        z=matched.iloc[-1]
        fields=['pass_attempts_pg','carries_pg'] if str(row['Position'])=='QB' else ['targets_pg','carries_pg']
        volume=sum(max(0.,_safe(z.get(f),0.)) for f in fields)
        if volume < .05:
            bad_rates.append(f"{row['Name']} ({row['TeamAbbrev']})")
    if bad_rates:
        raise ValueError('Stat baselines have no observed opportunity for: '+', '.join(bad_rates[:20])+
                         '. Check nflverse source schema/history and active roster; cannot safely simulate.')
    # Automated feeds can contain OUT/IR depth players with no observed-stat baseline.
    # They must still be excluded from the DFS pool, but there is no historical
    # opportunity to redistribute.  Only OUT players present in `rates` can
    # contribute missing targets/carries later in the team-allocation logic.
    rate_keys=set(zip(r['_name'].astype(str),r['team'].astype(str)))
    missing_out=out_keys-rate_keys
    redistributable_out=out_keys & rate_keys
    if any((n,t) in out_keys for n,t in zip(x['_name'],x['TeamAbbrev'])):
        raise ValueError('OUT player remained in eligible simulation pool')
    rm=r.drop_duplicates(['_name','team']).set_index(['_name','team'])
    ns=int(n_sims); rng=np.random.default_rng(seed)
    out=np.zeros((ns,len(x)),dtype=np.float32); stat_draws={}
    stat_draws['__MODEL_VERSION__']='Stat-driven V3.11: roster-independent team passing baseline'
    stat_draws['__BASELINE_RESOLUTION__']=baseline_report
    stat_draws['__INJURY_META__']={
        'out_players':sorted([f"{t}|{n}" for n,t in out_keys]),
        'out_count':len(out_keys),
        'redistributable_out_players':sorted([f"{t}|{n}" for n,t in redistributable_out]),
        'redistributable_out_count':len(redistributable_out),
        'zero_baseline_out_players':sorted([f"{t}|{n}" for n,t in missing_out]),
        'zero_baseline_out_count':len(missing_out),
    }
    name_index={(str(row['_name']),str(row['TeamAbbrev']),str(row['Position'])):i for i,row in x.iterrows()}
    teams=sorted(x['TeamAbbrev'].dropna().astype(str).unique())
    games=sorted(x['game'].dropna().astype(str).unique())
    processed=set()

    for game in games:
        gx=x[x.game.astype(str)==game]; gteams=list(dict.fromkeys(gx.TeamAbbrev.astype(str)))
        if len(gteams)!=2: continue
        # Shared pace/script shock for both teams in the game.
        pace=rng.lognormal(-0.5*.10**2,.10,size=ns)
        script=rng.normal(0,1,size=ns)
        for ti,team in enumerate(gteams):
            opp=gteams[1-ti]; tx=gx[gx.TeamAbbrev==team]
            qbrows=tx[tx.Position.astype(str)=='QB']
            # Team simulation must not disappear just because the active DK optimizer pool
            # has no eligible QB (injury/status/salary filtering can remove him).  Use the
            # best observed QB baseline for team-volume generation, while only assigning QB
            # DK points when an eligible QB is actually present in the pool.
            qbrow = None
            qname = ''
            qr = None
            if not qbrows.empty:
                qbrow=qbrows.sort_values('proj',ascending=False).iloc[0]
                qname=str(qbrow['_name'])
                qr=rm.loc[(qname,team)] if (qname,team) in rm.index else None
            if qr is None:
                team_qbs=r[(r['team'].astype(str)==team) & (r['position'].astype(str).str.upper()=='QB') & (~r['_name'].isin([n for n,t in out_keys if t==team]))].copy()
                if not team_qbs.empty:
                    team_qbs['_pa']=pd.to_numeric(team_qbs.get('pass_attempts_pg'),errors='coerce').fillna(0)
                    qr=team_qbs.sort_values(['_pa','games_played'],ascending=False).iloc[0]
                    if not qname:
                        qname=str(qr['_name'])
            # A salary file is not a complete team roster. Build team passing
            # volume from historical QB data for the TEAM, not only from the
            # highest-projected eligible DraftKings quarterback. When multiple
            # QBs have appeared, their per-game rates are weighted by starts/
            # games (not added together). This does not create a rostered QB.
            historical_qbs=r[(r['team'].astype(str)==team) &
                             (r['position'].astype(str).str.upper()=='QB')].copy()
            historical_qbs['_attempts']=pd.to_numeric(historical_qbs.get('pass_attempts_pg'),errors='coerce')
            historical_qbs['_games']=pd.to_numeric(historical_qbs.get('games_played'),errors='coerce').fillna(0).clip(lower=0)
            historical_qbs=historical_qbs[historical_qbs['_attempts'].notna() & (historical_qbs['_attempts']>0)]
            qb_eligible=not qbrows.empty
            raw_pass_pg=_safe(qr.get('pass_attempts_pg') if qr is not None else np.nan,32.0)
            if not historical_qbs.empty:
                # Game-weighting avoids treating a one-game replacement as the
                # team's full-season identity. Shrink low-sample baselines
                # toward a transparent league prior instead of hardcoding an
                # offense from one player or from the DK salary pool.
                w=historical_qbs['_games'].to_numpy(float)
                a=historical_qbs['_attempts'].to_numpy(float)
                if w.sum()>0:
                    team_pass_observed=float(np.average(a,weights=w))
                    sample_games=float(w.sum())
                else:
                    team_pass_observed=float(np.mean(a)); sample_games=0.
                reliability=sample_games/(sample_games+8.)
                team_pass_baseline=float(np.clip(reliability*team_pass_observed+(1-reliability)*32.,18.,48.))
            else:
                team_pass_observed=float('nan'); sample_games=0.
                team_pass_baseline=32.
            env=team_environment.get(team) or {}
            # A real, prior-game team-specific pass volume forecast now matters
            # to scoring.  Blend with the existing QB evidence rather than
            # abruptly replacing the QB's on-field baseline.
            if env:
                pass_pg=float(np.clip(.5*team_pass_baseline + .5*float(env['expected_pass_attempts']),18.,48.))
            else:
                pass_pg=team_pass_baseline
            stat_draws[f'__TEAM_PASS_BASELINE__|{team}']={
                'rostered_qb_available':qb_eligible,
                'historical_qb_count':int(len(historical_qbs)),
                'historical_qb_games':sample_games,
                'historical_qb_attempts_per_game':team_pass_observed,
                'selected_qb_attempts_per_game':raw_pass_pg,
                'team_pass_attempts_prior':32.,
                'team_pass_attempts_baseline':team_pass_baseline,
            }
            comp_pg=_safe(qr.get('completions_pg') if qr is not None else np.nan, pass_pg*.65)
            pass_yd_pg=_safe(qr.get('pass_yards_pg') if qr is not None else np.nan, 220.0)
            int_pg=_safe(qr.get('interceptions_pg') if qr is not None else np.nan, .65)
            sack_pg=_safe(qr.get('sacks_suffered_pg') if qr is not None else np.nan, 2.3)
            pass_td_pg=_safe(qr.get('passing_tds_pg') if qr is not None else np.nan, 1.45)

            skill=tx[tx.Position.astype(str).isin(['RB','WR','TE'])].copy()
            # Join observed opportunity; missing/depth players receive zero share unless no rates exist.
            rr=[]
            for _,p in skill.iterrows():
                key=(str(p['_name']),team); rr.append(rm.loc[key] if key in rm.index else None)
            target_rates=np.array([_safe(z.get('targets_pg') if z is not None else np.nan,0) for z in rr],float)
            carry_rates=np.array([_safe(z.get('carries_pg') if z is not None else np.nan,0) for z in rr],float)
            target_rates,carry_rates,snap_changes=apply_snap_roles_to_rates(
                target_rates,carry_rates,skill,snap_profiles)
            stat_draws[f'__SNAP_OPPORTUNITY__|{team}']={
                'observed_role_adjustments':snap_changes,
                'scored_player_count':sum(
                    (abs(d['target_after']-d['target_before'])>.0001 or
                     abs(d['carry_after']-d['carry_before'])>.0001)
                    for d in snap_changes),
                'source':'nflverse PFR actual offensive snaps; routes are unavailable',
            }
            # Historical player rates may span different depth charts and cannot
            # safely be summed into an unconstrained team rushing workload.
            # Derive a team-level play budget first, then allocate those plays.
            qb_carry_base=max(0.,_safe(qr.get('carries_pg') if qr is not None else np.nan,3.0))
            active_rush=float(np.maximum(carry_rates,0.).sum())+qb_carry_base
            # Include OUT players only to estimate the original team workload;
            # their carries will be redistributed once, not added as new plays.
            out_team=r[(r['team']==team)&(r['position'].astype(str).str.upper().isin(['RB','WR','TE']))].copy()
            out_team=out_team[[ (str(z['_name']),team) in redistributable_out for _,z in out_team.iterrows() ]] if redistributable_out else out_team.iloc[0:0]
            missing_rush=float(pd.to_numeric(out_team.get('carries_pg',pd.Series(dtype=float)),errors='coerce').fillna(0).clip(lower=0).sum())
            observed_rush=active_rush+missing_rush
            # A conservative team-play prior (65 plays) is used because the
            # input has no verified team-level plays/game feed. The 42-rush cap
            # prevents overlapping player samples from inventing 50+ rushes.
            inferred_rush=max(12.,65.-max(8.,pass_pg+sack_pg))
            team_rush=float(np.clip(.35*observed_rush+.65*inferred_rush,12.,42.))
            expected_dropbacks=float(np.clip(pass_pg+sack_pg,12.,55.))
            expected_plays=float(np.clip(expected_dropbacks+team_rush,48.,78.))
            if env:
                # The last-six-team-game historical volume replaces a generic
                # game-play prior, preserving the shared pass/rush budget.
                expected_plays=float(np.clip(float(env['expected_plays']),48.,78.))
                team_rush=float(np.clip(expected_plays-expected_dropbacks,12.,42.))
                expected_plays=float(np.clip(expected_dropbacks+team_rush,48.,78.))
            stat_draws[f'__TEAM_ENVIRONMENT__|{team}']={
                'scored_team_history':bool(env),
                'team_plays_prior':float(env['expected_plays']) if env else None,
                'team_pass_attempts_prior':float(env['expected_pass_attempts']) if env else None,
                'estimated_expected_plays':expected_plays,
                'estimated_pass_attempts':pass_pg,
                'historical_games':int(env['observed_games']) if env else 0,
                'source':str(env.get('source') or 'Historical QB volume; generic pace prior'),
            }

            # Script: positive means this team trails -> more passing, less rushing.
            # Sample one team play budget. Passing and rushing draw from the same
            # budget rather than independent Poissons that inflate total plays.
            team_script=script if ti==0 else -script
            # Conservative predictive bounds while a fitted team-pace feed is unavailable.
            # A Poisson/lognormal mixture previously produced 100+ offensive plays.
            # Truncation is a guardrail, NOT a historically fitted distribution.
            team_plays=np.clip(rng.poisson(expected_plays*pace),40,90)
            pass_fraction=np.clip(expected_dropbacks/max(expected_dropbacks+team_rush,1.),.28,.84)
            pass_fraction_scenario=np.clip(pass_fraction*np.exp(.12*team_script),.25,.88)
            dropbacks=rng.binomial(team_plays,pass_fraction_scenario)
            dropbacks=np.maximum(0,dropbacks)
            # DST pass-rush strength is conservatively inferred from its DFS baseline until
            # a dedicated team sack-rate table is available; QB sack tendency supplies offense side.
            dstrow=x[(x.TeamAbbrev==opp)&(x.Position.astype(str)=='DST')]
            dst_mean=_safe(dstrow.iloc[0].get('proj') if len(dstrow) else np.nan,6.5)
            dst_sack_mult=float(np.clip(1.0+(dst_mean-6.5)*.035,.78,1.28))
            sack_rate=np.clip((sack_pg/max(pass_pg+sack_pg,1))*dst_sack_mult,.025,.14)
            sacks=rng.binomial(dropbacks,sack_rate)
            attempts=np.maximum(0,dropbacks-sacks)
            rushes=team_plays-dropbacks
            stat_draws[f'__TEAM_PLAYS__|{team}']={
                'expected_plays':expected_plays,
                'observed_rush_rate_sum':observed_rush,
                'inferred_rush_prior':inferred_rush,
                'expected_rushes':team_rush,
                'simulated_plays':team_plays,
                'team_plays_guardrail_min':40,
                'team_plays_guardrail_max':90,
                'simulated_dropbacks':dropbacks,
                'simulated_rushes':rushes,
            }
            if not np.array_equal(team_plays,dropbacks+rushes):
                raise ValueError(f'V3.8 team-play conservation failed for {team}')

            # Target allocation from one team pass-attempt budget.
            # Separate confirmed-OUT historical usage from the genuine OTHER bucket.
            # OUT opportunity is redistributed exactly once.
            team_hist=r[(r['team']==team)&(r['position'].astype(str).str.upper().isin(['RB','WR','TE']))].copy()
            out_hist=team_hist[[ (str(z['_name']),team) in redistributable_out for _,z in team_hist.iterrows() ]] if redistributable_out else team_hist.iloc[0:0]
            missing_t=float(pd.to_numeric(out_hist.get('targets_pg',pd.Series(dtype=float)),errors='coerce').fillna(0).clip(lower=0).sum())
            missing_c=float(pd.to_numeric(out_hist.get('carries_pg',pd.Series(dtype=float)),errors='coerce').fillna(0).clip(lower=0).sum())
            leftover=max(pass_pg*.10, pass_pg-target_rates.sum()-missing_t, .5)
            carry_other=max(team_rush*.06,.5)
            pre_target_budget=float(target_rates.sum()+leftover+missing_t)
            pre_carry_budget=float(carry_rates.sum()+carry_other+missing_c)
            if missing_t>0 or missing_c>0:
                # Positional suitability matters: an unavailable RB's carries
                # predominantly move to available RBs, not to WRs/TEs.
                positions=skill['Position'].astype(str).str.upper().to_numpy()
                carry_role=np.where(positions=='RB',1.0,np.where(positions=='WR',.15,.05))
                target_role=np.where(positions=='WR',1.0,np.where(positions=='TE',.85,.75))
                target_rates,leftover=redistribute(target_rates,leftover,missing_volume=missing_t,role_weights=target_role)
                carry_rates,carry_other=redistribute(carry_rates,carry_other,missing_volume=missing_c,role_weights=carry_role)
            # Redistribution moves shares within each team budget; it must
            # never create extra attempts/carries or allocate negative volume.
            post_target_budget=float(target_rates.sum()+leftover)
            post_carry_budget=float(carry_rates.sum()+carry_other)
            if (not np.isclose(pre_target_budget,post_target_budget,rtol=0,atol=1e-7)
                or not np.isclose(pre_carry_budget,post_carry_budget,rtol=0,atol=1e-7)
                or np.any(target_rates<0) or np.any(carry_rates<0)
                or leftover<0 or carry_other<0):
                raise ValueError(f"Opportunity conservation failed for {team}; refusing simulated lineup output")
            stat_draws[f'__OPPORTUNITY__|{team}']={
                'target_budget_before':pre_target_budget,
                'target_budget_after':post_target_budget,
                'carry_budget_before':pre_carry_budget,
                'carry_budget_after':post_carry_budget,
                'removed_targets_pg':missing_t,'removed_carries_pg':missing_c,
                'active_target_rates_pg':dict(zip(skill['Name'].astype(str),target_rates.tolist())),
                'active_carry_rates_pg':dict(zip(skill['Name'].astype(str),carry_rates.tolist()))}

            tprob=np.append(np.maximum(target_rates,0),leftover); tprob=tprob/tprob.sum()
            targ=_multinomial_rows(rng,attempts,tprob)
            # Carry allocation includes QB and an unmodeled bucket.
            qb_carry=_safe(qr.get('carries_pg') if qr is not None else np.nan,3.0)
            cbase=np.append(np.maximum(carry_rates,0),[max(qb_carry,0),carry_other])
            cprob=cbase/cbase.sum(); carr=_multinomial_rows(rng,rushes,cprob)
            oth_carries=carr[:,-1].copy()

            team_rec_yd=np.zeros(ns); team_comp=np.zeros(ns); player_rec={}; player_rush={}
            for j,(_,p) in enumerate(skill.iterrows()):
                z=rr[j]; tt=targ[:,j]
                tr=max(_safe(z.get('targets_pg') if z is not None else np.nan,0),.1)
                recpg=_safe(z.get('receptions_pg') if z is not None else np.nan,tr*.62)
                catch_rate=float(np.clip(recpg/tr,.18,.92)); recs=rng.binomial(tt,catch_rate)
                ypr=max(2.0,_safe(z.get('receiving_yards_pg') if z is not None else np.nan,recpg*9.5)/max(recpg,.1))
                recyd=np.where(recs>0,rng.gamma(np.maximum(recs,1)*2.2,ypr/2.2)*rng.lognormal(-.5*.14**2,.14,ns),0.)
                cc=carr[:,j]
                cpg=max(_safe(z.get('carries_pg') if z is not None else np.nan,0),.1)
                ypc=_safe(z.get('rushing_yards_pg') if z is not None else np.nan,cpg*4.1)/cpg
                rushyd=np.where(cc>0,rng.normal(cc*ypc,np.sqrt(np.maximum(cc,1))*max(1.8,abs(ypc)*.75),ns),0.)
                player_rec[j]=(tt,recs,recyd); player_rush[j]=(cc,rushyd)
                team_rec_yd+=recyd; team_comp+=recs

            # Explicit accounting-only OTHER receiver bucket.
            oth_t=targ[:,-1]
            team_catch_rate=float(np.clip(comp_pg/max(pass_pg,1),.35,.82))
            oth_c=rng.binomial(oth_t,team_catch_rate)
            oth_y=np.where(oth_c>0,rng.gamma(np.maximum(oth_c,1)*2.0,4.5),0.)
            team_comp+=oth_c; team_rec_yd+=oth_y
            # Blend receiver-generated yards toward QB historical yards/attempt without breaking correlation.
            hist_ypa=pass_yd_pg/max(pass_pg,1); generated_ypa=team_rec_yd/np.maximum(attempts,1)
            scale=np.clip(hist_ypa/np.maximum(generated_ypa,.1),.72,1.35)
            team_rec_yd*=scale
            oth_y*=scale
            for j in player_rec: player_rec[j]=(player_rec[j][0],player_rec[j][1],player_rec[j][2]*scale)

            # Cache each scenario's completed passes; TD catches must be actual catches.
            rec_counts=np.column_stack([player_rec[j][1] for j in player_rec] + [oth_c])
            # Shared TD/turnover scoring uses historical rates only.
            rush_td_prior=max(.3,team_rush*.035)
            expected_td,fg_mean=_team_scoring_budget(pass_td_pg,rush_td_prior)
            # Tie touchdown opportunities to the *same simulated offensive plays*.
            # A binomial draw avoids the independent Poisson tail that previously
            # produced 10-11 offensive TDs while keeping a plausible high-scoring tail.
            # Eight is a conservative guardrail, not a claim of a fitted NFL maximum.
            td_probability=float(np.clip(expected_td/max(expected_plays,1.),.002,.095))
            total_td=np.minimum(rng.binomial(team_plays,td_probability),8)
            fgs=rng.poisson(fg_mean,size=ns)
            stat_draws[f'__TEAM_SCORING__|{team}']={
                'scoring_source':'historical_rates_only',
                'expected_offensive_tds':float(expected_td),
                'td_probability_per_team_play':td_probability,
                'expected_field_goals':float(fg_mean),
                'simulated_offensive_tds':total_td,
                'simulated_field_goals':fgs,
                'simulated_offensive_points':7*total_td+3*fgs,
            }
            pass_share=np.clip(pass_td_pg/max(pass_td_pg+rush_td_prior,.1),.35,.78)
            pass_td=rng.binomial(total_td,pass_share)
            # A passing touchdown must be a completed pass. Preserve the team
            # touchdown budget by moving any excess to rushing opportunities.
            # A TD must be backed by an actual reception or carry. Under a
            # very low-volume simulated game, a Poisson team scoring draw can
            # exceed the number of eligible scoring events. Constrain the draw
            # before allocating it instead of raising a capacity exception.
            total_capacity=team_comp.astype(np.int64)+np.asarray(rushes,dtype=np.int64)
            total_td=np.minimum(total_td,total_capacity)
            min_pass=np.maximum(0,total_td-np.asarray(rushes,dtype=np.int64))
            max_pass=np.minimum(total_td,team_comp.astype(np.int64))
            pass_td=np.clip(pass_td,min_pass,max_pass)
            rush_td=total_td-pass_td
            # All downstream team/DST audits must reflect the feasible TD draw.
            stat_draws[f'__TEAM_SCORING__|{team}']['simulated_offensive_tds']=total_td
            stat_draws[f'__TEAM_SCORING__|{team}']['simulated_offensive_points']=7*total_td+3*fgs
            int_rate=np.clip(int_pg/max(pass_pg,1),.005,.065); ints=rng.binomial(attempts,int_rate)

            # Allocate scores using *scenario-specific* opportunity rather than
            # fixed pregame target/carry shares. This preserves correlation:
            # a player drawing extra targets/carries in a simulated game also
            # earns more touchdown chances in that same game.
            # The rushing allocation explicitly includes the QB, previously
            # assigned zero rushing TDs regardless of his rushing workload.
            rec_td_full=np.zeros((ns,len(skill)+1),dtype=np.int16)
            rush_td_full=np.zeros((ns,len(skill)+2),dtype=np.int16)
            qb_carries=carr[:,-2]
            for si in range(ns):
                if pass_td[si]:
                    # Target share alone can assign a receiving TD to a player
                    # with zero catches or multiple TDs on one reception.
                    # Use actual catches as capacity and scoring opportunity.
                    catches=rec_counts[si]
                    weights=np.asarray(catches,dtype=float)+.05*(catches>0)
                    rec_td_full[si]=_td_allocation_with_capacity(
                        rng,int(pass_td[si]),weights,catches)
                if rush_td[si]:
                    # TD share is driven by both scenario carries and observed
                    # goal-line effectiveness; a QB's scramble share is not
                    # automatically his share of goal-line carries.
                    rush_weights=np.asarray(carr[si],dtype=float)
                    skill_tds=np.array([max(0.,_safe(z.get('rushing_tds_pg') if z is not None else 0,0)) for z in rr])
                    skill_hist=np.maximum(carry_rates,.25)
                    skill_factors=np.clip((skill_tds+.10)/(skill_hist*.065+.10),.55,2.0)
                    qb_td_pg=max(0.,_safe(qr.get('rushing_tds_pg') if qr is not None else 0,0))
                    # Empirical-Bayes shrinkage: limited rushing-TD observations must not
                    # turn ordinary QB carries into near-automatic goal-line TDs.
                    # Use a 4.5% rushing-TD-per-carry prior over 30 carries.
                    qb_games=max(1.,_safe(qr.get('games_played') if qr is not None else 0,1))
                    qb_hist_carries=max(0.,qb_carry)*qb_games
                    qb_hist_tds=qb_td_pg*qb_games
                    qb_smoothed_rate=(qb_hist_tds+30.*.045)/(qb_hist_carries+30.)
                    qb_factor=float(np.clip(qb_smoothed_rate/.065,.35,1.6))
                    rush_weights*=np.append(skill_factors,[qb_factor, .75])
                    # Only runners with carries in this simulation may score.
                    rush_weights=np.where(np.asarray(carr[si])>0,rush_weights,0.)
                    if rush_weights.sum()<=0: rush_weights=np.asarray(carr[si],dtype=float)
                    rush_weights/=rush_weights.sum()
                    rush_td_full[si]=_td_allocation_with_capacity(
                        rng,int(rush_td[si]),rush_weights,carr[si])
            rec_td_alloc=rec_td_full[:,:len(skill)]
            rush_td_alloc=rush_td_full[:,:len(skill)]
            qb_rush_td=rush_td_full[:,-2]
            oth_rec_td=rec_td_full[:,-1]
            oth_rush_td=rush_td_full[:,-1]

            # Explicit accounting-only OTHER rusher. Never enters DFS eligibility.
            modeled_ypc=[]
            for z in rr:
                c=max(_safe(z.get('carries_pg') if z is not None else np.nan,0),.1)
                modeled_ypc.append(_safe(z.get('rushing_yards_pg') if z is not None else np.nan,c*4.0)/c)
            oth_ypc=float(np.clip(np.nanmean(modeled_ypc) if modeled_ypc else 4.0,2.5,6.5))
            oth_rush_y=np.where(oth_carries>0,
                                np.maximum(0.,rng.normal(oth_carries*oth_ypc,
                                np.sqrt(np.maximum(oth_carries,1))*1.8,ns)),0.)

            stat_draws[f"__OTHER_RECEIVER__|{team}"]={
                'targets':oth_t,'receptions':oth_c,'receiving_yards':oth_y,'receiving_tds':oth_rec_td}
            stat_draws[f"__OTHER_RUSHER__|{team}"]={
                'carries':oth_carries,'rushing_yards':oth_rush_y,'rushing_tds':oth_rush_td}

            # Fail closed if team accounting does not conserve the simulated football totals.
            modeled_rec=np.sum([player_rec[j][1] for j in player_rec],axis=0) if player_rec else np.zeros(ns)
            modeled_rec_y=np.sum([player_rec[j][2] for j in player_rec],axis=0) if player_rec else np.zeros(ns)
            if not np.array_equal(team_comp,modeled_rec+oth_c):
                raise ValueError(f'V3.4 completion conservation failed for {team}')
            if not np.allclose(team_rec_yd,modeled_rec_y+oth_y,atol=.001):
                raise ValueError(f'V3.4 passing-yard conservation failed for {team}')
            if not np.array_equal(pass_td,np.sum(rec_td_alloc,axis=1)+oth_rec_td):
                raise ValueError(f'V3.4 passing-TD conservation failed for {team}')
            skill_carries=np.sum(carr[:,:len(skill)],axis=1) if len(skill) else np.zeros(ns,dtype=int)
            if not np.array_equal(rushes,skill_carries+carr[:,-2]+oth_carries):
                raise ValueError(f'V3.4 rush-attempt conservation failed for {team}')
            if not np.all(rec_td_full<=rec_counts):
                raise ValueError(f'V3.6 receiving TDs exceed catches for {team}')
            if not np.all(rush_td_full<=carr):
                raise ValueError(f'V3.6 rushing TDs exceed carries for {team}')
            if not np.array_equal(pass_td,np.sum(rec_td_full,axis=1)):
                raise ValueError(f'V3.6 passing TD conservation failed for {team}')
            if not np.array_equal(rush_td,np.sum(rush_td_alloc,axis=1)+qb_rush_td+oth_rush_td):
                raise ValueError(f'V3.4 rushing-TD conservation failed for {team}')

            # Record team passing even if DraftKings has no eligible QB for this
            # team. This is accounting only, never a selectable DFS player.
            # Receiver TDs and opposing DST scores depend on these team events.
            qidx=name_index.get((qname,team,'QB')) if qbrow is not None else None
            if qidx is None:
                stat_draws[f'__OTHER_QB_PASSING__|{team}']={
                    'pass_attempts':attempts,
                    'completions':team_comp,
                    'passing_yards':team_rec_yd,
                    'passing_tds':pass_td,
                    'interceptions':ints,
                    'sacks_taken':sacks,
                    'carries':carr[:,-2],
                    'rushing_tds':qb_rush_td,
                }
            # QB DK score.
            if qidx is not None:
                qb_carries=carr[:,-2]; qb_ypc=_safe(qr.get('rushing_yards_pg') if qr is not None else np.nan,15)/max(qb_carry,.5)
                qb_rushyd=np.where(qb_carries>0,rng.normal(qb_carries*qb_ypc,np.sqrt(np.maximum(qb_carries,1))*2.2,ns),0.)
                dk=.04*team_rec_yd+4*pass_td-ints+.1*qb_rushyd+6*qb_rush_td+3*(team_rec_yd>=300)+3*(qb_rushyd>=100)
                out[:,qidx]=dk.astype(np.float32)
                stat_draws[str(qbrow['Name'])]={'pass_attempts':attempts,'completions':team_comp,'passing_yards':team_rec_yd,'passing_tds':pass_td,'interceptions':ints,'sacks_taken':sacks,'carries':qb_carries,'rushing_yards':qb_rushyd,'rushing_tds':qb_rush_td}

            for j,(_,p) in enumerate(skill.iterrows()):
                idx=name_index.get((str(p['_name']),team,str(p['Position'])))
                if idx is None: continue
                tt,recs,recyd=player_rec[j]; cc,rushyd=player_rush[j]
                rtd=rec_td_alloc[:,j]; utd=rush_td_alloc[:,j]
                dk=recs+.1*recyd+6*rtd+.1*rushyd+6*utd+3*(recyd>=100)+3*(rushyd>=100)
                out[:,idx]=dk.astype(np.float32)
                stat_draws[str(p['Name'])]={'targets':tt,'receptions':recs,'receiving_yards':recyd,'receiving_tds':rtd,'carries':cc,'rushing_yards':rushyd,'rushing_tds':utd}

            # Opposing DST derives from this exact offensive scenario.
            didx_candidates=x.index[(x.TeamAbbrev==opp)&(x.Position.astype(str)=='DST')].tolist()
            if didx_candidates:
                didx=didx_candidates[0]
                # Fumble recoveries/defensive TDs are rare events tied partly to sacks/turnovers.
                fum=rng.binomial(np.maximum(sacks,0),.075)
                takeaways=ints+fum
                def_td=rng.binomial(np.minimum(takeaways,1),.075)
                saf=rng.binomial(1,.012,size=ns)
                pts_allowed=7*total_td+3*fgs+2*saf
                dst_dk=sacks+2*ints+2*fum+6*def_td+2*saf+_pa_points(pts_allowed)
                out[:,didx]=dst_dk.astype(np.float32)
                stat_draws[str(x.loc[didx,'Name'])]={'sacks':sacks,'interceptions':ints,'fumble_recoveries':fum,'defensive_tds':def_td,'safeties':saf,'points_allowed':pts_allowed}
            processed.add(team)

    # Verify that skill-player non-touchdown points were actually generated.
    # A previous broken deployment returned only 0/6/12 scores while passing
    # through the older mean>0 gate.  That result must never reach an optimizer.
    broken=[]
    for _, row in x[x['Position'].astype(str).isin(['RB','WR','TE'])].iterrows():
        name=str(row['Name'])
        stats=stat_draws.get(name)
        if stats is None:
            broken.append(f"{name}: no simulated football stats")
            continue
        non_td=(np.asarray(stats['receptions'],float)
                +.1*np.asarray(stats['receiving_yards'],float)
                +.1*np.asarray(stats['rushing_yards'],float))
        idx=name_index.get((str(row['_name']),str(row['TeamAbbrev']),str(row['Position'])))
        if idx is None:
            broken.append(f"{name}: missing lineup index")
            continue
        td=np.asarray(stats['receiving_tds'],float)+np.asarray(stats['rushing_tds'],float)
        if not np.allclose(out[:,idx],non_td+6*td
                           +3*(np.asarray(stats['receiving_yards'])>=100)
                           +3*(np.asarray(stats['rushing_yards'])>=100),atol=.0001):
            broken.append(f"{name}: DK scoring differs from simulated stats")
        elif float(np.mean(non_td))<0.15 and float(np.mean(td))>0.02:
            broken.append(f"{name}: TD-only output (non-TD mean={np.mean(non_td):.3f})")
    if broken:
        raise ValueError('V3.4 skill allocation validation failed: '
                         +'; '.join(broken[:18])
                         + '. No lineup CSV will be generated. Inspect nflverse baselines and team allocation.')

    # Fail closed on players that could not be generated rather than silently inventing points.
    means=out.mean(axis=0)
    missing=(means==0)&(~x.Position.astype(str).eq('DST'))&(pd.to_numeric(x.proj,errors='coerce').fillna(0)>1)
    if missing.any():
        names=x.loc[missing,'Name'].astype(str).tolist()[:15]
        raise ValueError('Stat-driven engine could not match observed-stat baselines for: '+', '.join(names))
    x['simulation_engine']='Stat-driven football simulation'
    x['portfolio_relevant']=True
    x['sim_p99_sanity']=np.percentile(out,99,axis=0); x['sim_max_sanity']=out.max(axis=0)
    return x,out,stat_draws
