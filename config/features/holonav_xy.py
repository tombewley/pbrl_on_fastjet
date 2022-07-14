def x(tr, _): return tr[:,4]
def y(tr, _): return tr[:,5]
P = {
    "pbrl": {
        "model": {
            "featuriser": {
                "features": [x, y]
            }
        }
    }
}
