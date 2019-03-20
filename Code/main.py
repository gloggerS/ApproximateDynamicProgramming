import re

t = "t[80]"

pat = r".*?\[(.*)\].*"
match = re.search(pat, t)
match.group(1)