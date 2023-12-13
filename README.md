# CS_646_project 
Personalization of Language model using Psuedorelevance feedback

command to run eval - please change the path to the file as needed

Evaluation of Tweets dataset LaMP-4:

python3 CS_646_project/src/eval/eval_task.py --golds_json CS_646_project/src/tweet/tweet_golds.json --preds_json CS_646_project/src/tweet/LLMpreds_tweet.txt --task_name LaMP_7 --output_file  CS_646_project/src/tweet/results_tweet.txt

Evaluation of News dataset LaMP-7 : 
 python3 CS_646_project/src/eval/eval_task.py --golds_json CS_646_project/src/news/golds_json_news.json --preds_json CS_646_project/src/news/Preds_json.json --task_name LaMP_4 --output_file  CS_646_project/src/news/results_news.txt