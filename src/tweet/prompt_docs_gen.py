import json
import pandas as pd
import pyterrier as pt
import re
import shutil, os
import warnings
warnings.filterwarnings('ignore')
#uncomment for first time index creation
#pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

with open("/content/sample_data/dev_questions-3.json","r") as file:
    data=json.load(file)
    #print("length:",len(data))
    usercount=len(data)
    for i in range(len(data)):
        print("iteration:",i)
        tot_df=pd.DataFrame.from_dict(data[i])
        user_df=pd.DataFrame.from_dict(data[i]['profile'])
        user_df=user_df.rename(columns={"id": "docno"})
        #print(user_df)
        query_df=tot_df[['id','input']]
        query_df=query_df.drop_duplicates(subset=['id','input'])
        query_df=query_df.rename(columns={"id": "qid","input":"query"})

        query_df['query']=[re.sub('\W+',' ', str(x)) for x in query_df['query']]

        #print(query_df['query'].to_string())
        #print("query_df",query_df)

        #To remove indexd folder
        if os.path.exists('/content/pd_index_0'):
          for i in range(1):
            temp_folder_path="./pd_index_"+str(i)
            shutil.rmtree(temp_folder_path)

        temp_index_path="./pd_index_"+str(i)
        pd_indexer = pt.DFIndexer(temp_index_path,stopwords=None)
        #uncomment for first time index creation
        indexref = pd_indexer.index(user_df["text"], user_df["docno"])
        index = pt.IndexFactory.of("/content/"+temp_index_path+"/data.properties")
        #print(index.getCollectionStatistics())

        #scoring with bm25 and rm3 -> bm25
        bm25 = pt.BatchRetrieve(index, wmodel="BM25")
        rm3_pipe = bm25 >> pt.rewrite.RM3(index,fb_terms=500, fb_docs=15, fb_lambda=0.60)
        new_qdf = rm3_pipe.transform(query_df) #replacing df with large_df
        new_qdf['query']=new_qdf['query'].apply(lambda x: re.sub('\W+',' ', str(x)))
        new_qdf['query']=new_qdf['query'].str.replace('\d+', '')
        new_qdf['merged_query'] = new_qdf.apply(lambda row: row['query_0']+ ' ' + row['query'], axis=1)
        new_qdf = new_qdf.drop(['query', 'query_0'], axis=1)
        new_qdf = new_qdf.rename(columns={'merged_query': 'query'})
        prf_df=bm25.transform(new_qdf)
        prf_df=prf_df.head(3)
        prf_df_docs=prf_df['docno']
        prompt_docs = pd.merge(user_df, prf_df_docs, on='docno', how='inner')
        print("prompt_docs",prompt_docs)
        dic = query_df.to_dict()
        qid = (dic['qid'][0])
        print(qid)
        df = pd.merge(user_df, prompt_docs, on='docno')
        df = df.drop(df.columns[-1:],axis=1)
        first_prompt_doc_text = df.to_dict('records')[0]['text_x']
        original_query = query_df.to_dict('records')[0]['query']
        input_to_LLM = original_query
        print(input_to_LLM)
        with open('/content/sample_data/output.txt', 'a') as f:
          f.write(qid  + '-' + input_to_LLM + '\n')
        