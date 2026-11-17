"""Research-only official NFL news-hub discovery smoke; never creates statuses."""
import argparse,json
from ai2_nfl_official_article_discovery import discover_official_links


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--week",type=int,default=11)
    a=p.parse_args()
    result=discover_official_links(week=a.week)
    print(json.dumps(result,indent=2,sort_keys=True))
    if any(x.get("kind")=="OFFICIAL_INACTIVES" for x in result["articles"]):
        print("DISCOVERED an official NFL.com inactive URL; article still needs source verification")
    else:
        print("NO OFFICIAL INACTIVES linked by monitored publisher hubs; no player status inferred")


if __name__=="__main__":
    main()
