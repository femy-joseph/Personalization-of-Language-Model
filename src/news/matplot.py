import matplotlib.pyplot as plt


def display_graph(x,y):
        #plotting the graph for both testing and training
        plt.plot(x, y,marker="o",label = "With PRF", linestyle="-",color='orange')
        #plt.xscale('log')
        plt.xlabel("K(no. of user profiles)")  # add X-axis label
        plt.ylabel("Rouge-L score")  # add Y-axis label
        plt.title("Rouge-L score over K  : News Headline Generation")  # add title

        plt.show()


#rouge-1 score Tweets =[0.46841569699990635,0.34344555,0.184553,0.1600034]
#rouge1 news = [0.1431569699990635,0.155587,0.171943,0.1692034] 

d=[0,1,2,3]

score=[0.1391569699990635,0.145587,0.15143,0.148034] 
x=d 
y=score 
display_graph(x,y,)