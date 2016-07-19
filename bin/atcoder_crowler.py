# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import urllib
from bs4 import BeautifulSoup

rootpath = os.path.join(os.path.dirname(__file__), "..")

parser = argparse.ArgumentParser()
parser.add_argument("contest", help="e.g. abc041")
parser.add_argument("stage", help="e.g. a")
parser.add_argument("language", help="e.g. python2_2.7.6")
parser.add_argument("status", help="e.g. AC")
args = parser.parse_args()

datadir = os.path.join(rootpath, "data", args.contest, args.stage, args.language, args.status)
if not os.path.exists(datadir):
    os.makedirs(datadir)

site = "http://{contest}.contest.atcoder.jp".format(contest=args.contest)
def save_submissions(url):
    soup = BeautifulSoup(urllib.urlopen(url), "html.parser")
    if soup.find("table") is None:
        return False
    submissions_soup = soup.find("table").find("tbody")
    for item in submissions_soup.find_all("tr"):
        submission_url = item.find_all("td")[-1].find("a")["href"]
        submission_id = os.path.basename(submission_url)

        submission_soup = BeautifulSoup(urllib.urlopen(site+submission_url), "html.parser")
        code = submission_soup.find("pre").text

        filepath = os.path.join(datadir, submission_id)
        with open(filepath, "w+") as f:
            try:
                f.write(code)
            except:
                pass
        print(filepath)
    return True

pages = 10
for page in range(1, pages+1):
    url = "http://{contest}.contest.atcoder.jp/submissions/all/{page}?task_screen_name={contest}_{stage}&language_screen_name={language}&status={status}".format(contest=args.contest, stage=args.stage, language=args.language, status=args.status, page=page)
    print(url)
    save_submissions(url)
