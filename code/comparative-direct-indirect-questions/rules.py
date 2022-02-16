import re

# Rules to classify indirect questions
pConj1 = re.compile(' or | vs| versus|between | from | over | and | than ', re.IGNORECASE)
p1 = re.compile('JJS|RBS')
p2 = re.compile(r"(?=.*who )(?=.* first)", re.IGNORECASE)
p3 = re.compile(r"what are good|the top|what are some|advantages of|benefits of", re.IGNORECASE)
p4 = re.compile("advantages and disadvantages of")

# Rules to classify direct questions
pConj2 = re.compile(' or | vs| versus|between | from | over ', re.IGNORECASE)
p6 = re.compile('JJR|RBR')
p7 = re.compile(r"(?=.*better)(?=.* or )|(?=.*better)(?=.* than )|(?=.*better)(?=.* vs)|(?=.*better)(?=.* versus)|(?=.*better)(?=.* and )\
|(?=.*better)(?=.*between )|(?=.*best)(?=.* or )|(?=.*first)(?=.* or )", re.IGNORECASE)
p8 = re.compile(r"(?=.*diff)(?=.*between )|(?=.*diff)(?=.* vs)|(?=.*diff)(?=.* versus)\
|(?=.*diff)(?=.* from )|(?=.*diff)(?=.* to )", re.IGNORECASE)
p8_1 = re.compile(r"(?=.*diff)(?=.* or )", re.IGNORECASE)
p9 = re.compile(r"(?=.*same)(?=.* and )|(?=.*same)(?=.* or )|(?=.*same)(?=.* as )|(?=.*similar)(?=.* or )|(?=.*similar)(?=.* and )\
|(?=.*compare)(?=.* to )|(?=.*compare)(?=.* with )|(?=.*prefer)(?=.* over )|(?=.*prefer)(?=.* to )|(?=.*which)(?=.* correct)|(?=.*how )(?=.* correct)\
|(?=.*correct)(?=.* or )", re.IGNORECASE)
p10 = re.compile(r"(?=.*distinguish)(?=.* from )|(?=.*replace)(?=.* with )", re.IGNORECASE)

def p5(q):
    if 'advantages and disadvantages' in q:
        if pConj1.search(q.split('advantages and disadvantages')[1]): return True
        
# Additional rules used in the rest function
pConj3 = re.compile('vs| versus', re.IGNORECASE)        

# Returns True if a question indirect, False otherwise
def indirect(q, p):
    if ((p1.search(p) or p2.search(q) or p3.search(q) or p4.search(q)) and not pConj1.search(q))\
        and 'compar' not in q and 'superior to' not in q and 'is fake' not in q and 'the same as' not in q and 'similar to' not in q\
        and 'as good as' not in q and 'as bad as' not in q and not p5(q) and not 'which' in q: 
        return True
    else: return False
    
# Returns True if a question direct, False otherwise
def direct(q, p):
    if (p6.search(p) and not p1.search(p) and pConj2.search(q)) or p7.search(p) or p8.search(q) or (p8_1.search(q) and p.find('diff') < p.find('or'))\
        or (p9.search(q) and not 'to compare' in q) or p10.search(q) or ('prefer' in q and pConj2.search(q))\
        or ('should' in q and pConj2.search(q)) or ('distinguish' in q and pConj2.search(q))\
        or ('win' in q and pConj2.search(q)) or ('won' in q and pConj2.search(q)):
        return True
    else: return False

# Applied after the direct and indirect functions to increase recall.
# True if a question direct, False otherwise
def rest(q, p):
    if (pConj3.search(q) or p7.search(q)) and not p3.search(q) and 'most' not in q and 'best' not in q:
        return True
    else: return False
