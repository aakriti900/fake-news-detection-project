"""Generates a small demo dataset (data/news.csv) so the project runs out of the box.
For real results, replace it with the Kaggle 'Fake and Real News' dataset (see README)."""
import random
import pandas as pd

random.seed(42)

real_templates = [
    "{org} announces {topic} plan after meeting with officials, report says",
    "Government releases {topic} figures for the {period}, showing {trend}",
    "{org} study finds {trend} in {topic} over the past {period}",
    "Parliament passes bill on {topic} following weeks of debate",
    "Central bank holds rates steady as {topic} data shows {trend}",
    "Researchers at {org} publish peer-reviewed findings on {topic}",
    "Officials confirm {topic} review will begin next {period}, spokesperson says",
    "{org} reports quarterly earnings in line with analyst expectations on {topic}",
]
fake_templates = [
    "SHOCKING: {org} secretly hiding the truth about {topic}, insiders reveal!!!",
    "You won't believe what {org} just did about {topic} - they don't want you to know",
    "BREAKING: Miracle {topic} cure banned by {org}, doctors hate this one trick",
    "Anonymous source EXPOSES {org} conspiracy over {topic}, mainstream media silent",
    "{org} caught lying about {topic}, share before they delete this!",
    "Scientists STUNNED: {topic} will destroy everything by next {period}, government cover-up",
    "Celebrity reveals {org} plot to control {topic} - the shocking truth finally out",
    "100% PROOF that {topic} is a hoax created by {org}, wake up people",
]
orgs = ["The Health Ministry", "The World Bank", "University researchers", "The Finance Department",
        "Big Pharma", "The Election Commission", "Global elites", "The Energy Agency", "The Tech Giant"]
topics = ["inflation", "vaccination", "climate", "unemployment", "elections", "energy prices",
          "education reform", "healthcare", "cryptocurrency", "water supply"]
periods = ["month", "quarter", "year", "week"]
trends = ["a modest increase", "a slight decline", "steady growth", "mixed results"]

def fill(t):
    return t.format(org=random.choice(orgs), topic=random.choice(topics),
                    period=random.choice(periods), trend=random.choice(trends))

rows = []
for _ in range(400):
    rows.append((fill(random.choice(real_templates)), "REAL"))
    rows.append((fill(random.choice(fake_templates)), "FAKE"))
random.shuffle(rows)
pd.DataFrame(rows, columns=["text", "label"]).to_csv("data/news.csv", index=False)
print("Created data/news.csv with", len(rows), "rows")
