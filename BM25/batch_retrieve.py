import pyterrier as pt
import json
pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

with open("/content/sample_data/profile_texts.json","r") as file:
    profiles=json.load(file)

iter_indexer = pt.IterDictIndexer("/content/Lamp4_index2",meta={'docno': 20, 'text': 4096})
indexref1 = iter_indexer.index(profiles)

index = pt.IndexFactory.of("/content/Lamp4_index2/data.properties")
print(index.getCollectionStatistics())

#rm3 = pt.rewrite.RM3("/content/Lamp4_index2")

bm25 = pt.BatchRetrieve("/content/Lamp4_index2", wmodel="BM25")
bm25.transform("generate headline for the article Here are a few tips to keep your")