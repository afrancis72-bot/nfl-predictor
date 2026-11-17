"""Read-only HTML structure diagnostic; prints tag paths, NEVER source status data."""
from bs4 import BeautifulSoup
from ai2_nfl_official_inactives_capture import fetch_official_html,_team_heading
url="https://www.nfl.com/news/nfl-week-10-inactives-players-ruled-out-sunday-2026-season"
raw=fetch_official_html(url)
soup=BeautifulSoup(raw,"html.parser")
print("nbytes",len(raw),"articles",len(soup.select("article")),
      "h3",len(soup.find_all("h3")),"h2",len(soup.find_all("h2")),
      "li",len(soup.find_all("li")),"div",len(soup.find_all("div")))
print("title",soup.title.get_text(" ",strip=True)[:145] if soup.title else "none")
for h in soup.find_all(["h2","h3","h4"]):
    t=h.get_text(" ",strip=True)
    team=_team_heading(t)
    if team in ("KC","ATL") or ("early" in t.lower() and len(t)<50):
        print("heading",h.name,repr(t[:85]),"team",team,"path",
              "/".join(
                  [f"{n.name}[{','.join(n.get('class',[])[:2])}]" 
                   for n in list(h.parents)[:9] if getattr(n,"name",None)]))
for selector in (
    "article",".nfl-c-article__body",".nfl-c-article__content",
    ".nfl-c-article__container","main","main article","[role=main]"):
    nodes=soup.select(selector)
    print("SELECTOR",selector,"number",len(nodes),"team_headings",[
        [_team_heading(x.get_text(" ",strip=True)) for x in node.find_all("h3")
         if _team_heading(x.get_text(" ",strip=True)) in ("KC","ATL")]
        for node in nodes[:3]])
