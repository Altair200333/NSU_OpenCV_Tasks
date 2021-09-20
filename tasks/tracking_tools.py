
def getHits(trainKeypoints, ids):
    hits = []
    for match in ids:
        hits.append(trainKeypoints[match])
    return hits
