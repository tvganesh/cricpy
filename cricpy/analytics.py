import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from pylab import rcParams

##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 11 Oct 2018
# Function: batsman4s
# This function plots the number of 4s vs the runs scored in the innings by the batsman
#

###########################################################################################
def batsman4s(file, name="A Hookshot"):
    '''
    Plot the numbers of 4s against the runs scored by batsman
    
    Description
    
    This function plots the number of 4s against the total runs scored by batsman. A 2nd order polynomial regression curve is also plotted. The predicted number of 4s for 50 runs and 100 runs scored is also plotted
    
    Usage
    
    batsman4s(file, name="A Hookshot")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    
    Note

    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsman6s
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    tendulkar  = getPlayerData(35320,dir="../",file="tendulkar.csv",type="batting")
    homeOrAway=[1,2],result=[1,2,4]
    batsman4s("tendulkar.csv", "Sachin Tendulkar")
    
    '''   
    # Clean the batsman file and create a complete data frame
    df = clean(file)
    df['Runs'] = pd.to_numeric(df['Runs'])
    df['4s'] = pd.to_numeric(df['4s'])
    
    df1 = df[['Runs','4s']].sort_values(by=['Runs'])

    
    # Set figure size
    rcParams['figure.figsize'] = 10,6
    
    # Get numnber of 4s and runs scored
    runs = pd.to_numeric(df1['Runs'])
    x4s = pd.to_numeric(df1['4s'])
     
    atitle = name + "-" + "Runs scored vs No of 4s" 
    
    # Plot no of 4s and a 2nd order curve fit   
    plt.scatter(runs, x4s, alpha=0.5)
    plt.xlabel('Runs')
    plt.ylabel('4s')
    plt.title(atitle)
    
    # Create a polynomial of degree 2
    poly = PolynomialFeatures(degree=2)
    runsPoly = poly.fit_transform(runs.reshape(-1,1))
    linreg = LinearRegression().fit(runsPoly,x4s)
    
    plt.plot(runs,linreg.predict(runsPoly),'-r')
    
    # Predict the number of 4s for 50 runs
    b=poly.fit_transform((np.array(50)))
    c=linreg.predict(b)
    plt.axhline(y=c, color='b', linestyle=':')
    plt.axvline(x=50, color='b', linestyle=':')
    
    
    # Predict the number of 4s for 100 runs
    b=poly.fit_transform((np.array(100)))
    c=linreg.predict(b)
    plt.axhline(y=c, color='b', linestyle=':')
    plt.axvline(x=100, color='b', linestyle=':')
    
    plt.text(180, 0.5,'Data source-Courtesy:ESPN Cricinfo',
         horizontalalignment='center',
         verticalalignment='center',
         )
    plt.show()
    plt.gcf().clear()
    return

   
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 13 Oct 2018
# Function: batsman6s
# This function plots the number of 6s vs the runs scored in the innings by the batsman
#
###########################################################################################

def batsman6s(file, name="A Hookshot") :
    '''
    Description

    Compute and plot the number of 6s in the total runs scored by batsman
    
    Usage
    
    batsman6s(file, name="A Hookshot")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    
    Examples
    # Get or use the <batsman>.csv obtained with getPlayerData()
    # tendulkar = getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    # batsman6s("tendulkar.csv","Sachin Tendulkar")

    '''
    x6s = []
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6
    
    # Clean the batsman file and create a complete data frame
    df = clean (file)  
    
    # Remove all rows where 6s are 0
    a= df['6s'] !=0
    b= df[a]
    
    x6s=b['6s'].astype(int)
    runs=pd.to_numeric(b['Runs'])
    
    # Plot the 6s as a boxplot
    atitle =name + "-" + "Runs scored vs No of 6s" 
    df1=pd.concat([runs,x6s],axis=1)
    fig = sns.boxplot(x="6s", y="Runs", data=df1)
    plt.title(atitle)
    plt.text(2.2, 10,'Data source-Courtesy:ESPN Cricinfo',
         horizontalalignment='center',
         verticalalignment='center',
         )
    plt.show()
    plt.gcf().clear()
    return
    
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 14 Oct 2018
# Function: batsmanAvgRunsGround
# This function plots the average runs scored by batsman at the ground. The xlabels indicate
# the number of innings at ground
#
###########################################################################################

def batsmanAvgRunsGround(file, name="A Latecut"):
    '''
    Description
    
    This function computed the Average Runs scored on different pitches and also indicates the number of innings played at these venues
    
    Usage
    
    batsmanAvgRunsGround(file, name = "A Latecut")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanDismissals, batsmanMovingAverage, batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    ##tendulkar  = getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=[1,2],result=[1,2,4])
    batsmanAvgRunsGround("tendulkar.csv","Sachin Tendulkar")  
    
    '''
    
    batsman = clean(file)
    rcParams['figure.figsize'] = 10,6
    batsman['Runs']=pd.to_numeric(batsman['Runs'])
    
    # Aggregate as sum, mean and count
    df=batsman[['Runs','Ground']].groupby('Ground').agg(['sum','mean','count'])
    
    #Flatten multi-levels to column names
    df.columns= ['_'.join(col).strip() for col in df.columns.values]
    
    # Reset index
    df1=df.reset_index(inplace=False)
    
    atitle = name + "'s Average Runs at Ground"
    plt.xticks(rotation="vertical",fontsize=8)
    plt.axhline(y=50, color='b', linestyle=':')
    plt.axhline(y=100, color='r', linestyle=':')
    
    ax=sns.barplot(x='Ground', y="Runs_mean", data=df1)
    plt.title(atitle)
    plt.text(30, 180,'Data source-Courtesy:ESPN Cricinfo',\
             horizontalalignment='center',\
             verticalalignment='center',\
             )
    plt.show()
    plt.gcf().clear()
    return

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 14 Oct 2018
# Function: batsmanAvgRunsOpposition
# This function plots the average runs scored by batsman versus the opposition. The xlabels indicate
# the Opposition and the number of innings at ground
#
###########################################################################################

def batsmanAvgRunsOpposition(file, name="A Latecut"):
    '''
    This function computes and plots the Average runs against different opposition played by batsman
    
    Description
    
    This function computes the mean runs scored by batsman against different opposition
    
    Usage
    
    batsmanAvgRunsOpposition(file, name = "A Latecut")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanDismissals, batsmanMovingAverage, batsmanPerfBoxHist batsmanAvgRunsGround
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkar = getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=[1,2],result=[1,2,4])
    batsmanAvgRunsOpposition("tendulkar.csv","Sachin Tendulkar")
'''
    batsman = clean(file)    
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6
    
    batsman['Runs']=pd.to_numeric(batsman['Runs'])
    
    # Aggregate as sum, mean and count
    df=batsman[['Runs','Opposition']].groupby('Opposition').agg(['sum','mean','count'])
    
    #Flatten multi-levels to column names
    df.columns= ['_'.join(col).strip() for col in df.columns.values]
    
    # Reset index
    df1=df.reset_index(inplace=False)
    
    atitle = name + "'s Average Runs vs Opposition"
    plt.xticks(rotation="vertical",fontsize=8)

    ax=sns.barplot(x='Opposition', y="Runs_mean", data=df1)
    plt.axhline(y=50, color='b', linestyle=':')
    plt.title(atitle)
    plt.text(5, 50, 'Data source-Courtesy:ESPN Cricinfo',\
             horizontalalignment='center',\
             verticalalignment='center',\
             )
    plt.show()
    plt.gcf().clear()
    return

    
    
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: batsmanContributionWonLost
# This plots the batsman's contribution to won and lost matches
#
###########################################################################################

def batsmanContributionWonLost(file,name="A Hitter"):
    '''
    Display the batsman's contribution in matches that were won and those that were lost
    
    Description
    
    Plot the comparative contribution of the batsman in matches that were won and lost as box plots
    
    Usage
    
    batsmanContributionWonLost(file, name = "A Hitter")
    Arguments
    
    file	
    CSV file of batsman from ESPN Cricinfo obtained with getPlayerDataSp()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanMovingAverage batsmanRunsPredict batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkarsp = getPlayerDataSp(35320,".","tendulkarsp.csv","batting")
    
    batsmanContributionWonLost("tendulkarsp.csv","Sachin Tendulkar")
    '''
    playersp = clean(file)
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6
    
    # Create a column based on result
    won = playersp[playersp['result'] == 1]
    lost = playersp[(playersp['result']==2) | (playersp['result']==4)]
    won['status']="won"
    lost['status']="lost"
    
    # Stack dataframes
    df= pd.concat([won,lost])
    df['Runs']=  pd.to_numeric(df['Runs'])
    
    ax = sns.boxplot(x='status',y='Runs',data=df)
    atitle = name + "-" + "- Runs in games won/lost-drawn" 
    plt.title(atitle)
    plt.text(0.5, 200,'Data source-Courtesy:ESPN Cricinfo',
         horizontalalignment='center',
         verticalalignment='center',
         )
    plt.show()
    plt.gcf().clear()
    return

    

import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 17 Oct 2018
# Function: batsmanCumulativeAverageRuns
# This function computes and plots the cumulative average runs by a batsman
#
###########################################################################################
def batsmanCumulativeAverageRuns(file,name="A Leg Glance"):
    '''
    Batsman's cumulative average runs
    
    Description
    
    This function computes and plots the cumulative average runs of a batsman
    
    Usage
    
    batsmanCumulativeAverageRuns(file,name= "A Leg Glance")
    Arguments
    
    file	
    Data frame
    name	
    Name of batsman
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanCumulativeStrikeRate bowlerCumulativeAvgEconRate bowlerCumulativeAvgWickets
    
    Examples
    
    batsmanCumulativeAverageRuns("tendulkar.csv", "Sachin Tendulkar")
    '''

    batsman= clean(file) 
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6
    
    runs=pd.to_numeric(batsman['Runs'])
    
    # Compute cumulative average
    cumAvg = runs.cumsum()/pd.Series(np.arange(1, len(runs)+1), runs.index)
    
    atitle = name + "- Cumulative Average vs No of innings"
    plt.plot(cumAvg)
    plt.xlabel('Innings')
    plt.ylabel('Cumulative average')
    plt.title(atitle)
    plt.text(200,20,'Data source-Courtesy:ESPN Cricinfo',
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
    plt.show()
    plt.gcf().clear()
    return
    
import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 17 Oct 2018
# Function: batsmanCumulativeStrikeRate
# This function computes and plots the cumulative average strike rate of a batsman
#
###########################################################################################

def batsmanCumulativeStrikeRate(file,name="A Leg Glance"):  
    '''
    Batsman's cumulative average strike rate
    
    Description
    
    This function computes and plots the cumulative average strike rate of a batsman
    
    Usage
    
    batsmanCumulativeStrikeRate(file,name= "A Leg Glance")
    Arguments
    
    file	
    Data frame
    name	
    Name of batsman
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanCumulativeAverageRuns bowlerCumulativeAvgEconRate bowlerCumulativeAvgWickets
    
    Examples
    
    ## Not run: 
    batsmanCumulativeStrikeRate("tendulkar.csv", "Sachin Tendulkar")
    '''

    batsman= clean(file) 
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6
    
    strikeRate=pd.to_numeric(batsman['SR'])
    
    # Compute cumulative strike rate
    cumStrikeRate = strikeRate.cumsum()/pd.Series(np.arange(1, len(strikeRate)+1), strikeRate.index)
    
    atitle = name + "- Cumulative Strike rate vs No of innings"

    plt.xlabel('Innings')
    plt.ylabel('Cumulative Strike Rate')
    plt.title(atitle)
    plt.plot(cumStrikeRate)
    
    plt.text(200,60,'Data source-Courtesy:ESPN Cricinfo',
                 horizontalalignment='center',
                 verticalalignment='center',
                 ) 
    plt.show()
    plt.gcf().clear()
    return
    

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 13 Oct 2018
# Function: batsman6s
# This function plots the batsman dismissals
#
###########################################################################################

def batsmanDismissals(file, name="A Squarecut"):
    '''
    Display a 3D Pie Chart of the dismissals of the batsman
    
    Description
    
    Display the dismissals of the batsman (caught, bowled, hit wicket etc) as percentages
    
    Usage
    
    batsmanDismissals(file, name="A Squarecut")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanMeanStrikeRate, batsmanMovingAverage, batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkar= getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    
    batsmanDismissals("tendulkar.csv","Sachin Tendulkar")
    '''

    batsman = clean(file)

    # Set figure size
    rcParams['figure.figsize'] = 10,6
    
    d = batsman['Dismissal']     
    # Convert to data frame
    df = pd.DataFrame(d)
    
    df1=df['Dismissal'].groupby(df['Dismissal']).count()
    df2 = pd.DataFrame(df1)
    df2.columns=['Count']
    df3=df2.reset_index(inplace=False)
    
    # Plot a pie chart
    plt.pie(df3['Count'], labels=df3['Dismissal'],autopct='%.1f%%')
    atitle = name +  "-Pie chart of dismissals"
    plt.suptitle(atitle, fontsize=16)
    plt.show()
    plt.gcf().clear()
    return
    

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 13 Oct 2018
# Function: batsmanMeanStrikeRate
# This function plot the Mean Strike Rate of the batsman against Runs scored as a continous graph
#
###########################################################################################

def batsmanMeanStrikeRate(file, name="A Hitter"):
    '''
    batsmanMeanStrikeRate {cricketr}	R Documentation
    Calculate and plot the Mean Strike Rate of the batsman on total runs scored
    
    Description
    
    This function calculates the Mean Strike Rate of the batsman for each interval of runs scored
    
    Usage
    
    batsmanMeanStrikeRate(file, name = "A Hitter")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanDismissals, batsmanMovingAverage, batsmanPerfBoxHist batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkar  <- getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    batsmanMeanStrikeRate("tendulkar.csv","Sachin Tendulkar")
    '''
    batsman = clean(file)
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6    
    
    runs= pd.to_numeric(batsman['Runs'])

    # Create the histogram 
    hist, bins = np.histogram(runs, bins = 20)
    midBin=[]
    SR=[]
    
    # Loop through 
    for i in range(1,len(bins)):
        # Find the mean of the bins (Runs)
        midBin.append(np.mean([bins[i-1],bins[i]]))
        
        # Filter runs that are are between 2 bins
        batsman['Runs']=pd.to_numeric(batsman['Runs'])
        a=(batsman['Runs'] > bins[i-1]) & (batsman['Runs'] <= bins[i])
        df=batsman[a]
        SR.append(np.mean(df['SR']))
        
        atitle = name + "-" + "Strike rate in run ranges" 
    
        # Plot no of 4s and a 2nd order curve fit   
        plt.scatter(midBin, SR, alpha=0.5)
        plt.plot(midBin, SR,color="r", alpha=0.5)
        plt.xlabel('Runs')
        plt.ylabel('Strike Rate')
        plt.title(atitle)
        plt.text(180, 50,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             )
    plt.show()
    plt.gcf().clear()
    return


import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 17 Oct 2018
# Function: batsmanMovingAverage
# This function computes and plots the Moving Average of the batsman across his career
#
###########################################################################################

# Compute a moving average
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def batsmanMovingAverage(file,name="A Squarecut") :
    '''
    Calculate and plot the Moving Average of the batsman in his career
    
    Description
    
    This function calculates and plots the Moving Average of the batsman in his career
    
    Usage
    
    batsmanMovingAverage(file,name="A Squarecut")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanDismissals, batsmanMeanStrikeRate, batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkar  <- getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    
    batsmanMovingAverage("tendulkar.csv","Sachin Tendulkar")
    '''
    # Compute the moving average of the time series
    batsman = clean(file)
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6   
    runs=pd.to_numeric(batsman['Runs'])
    
    date= pd.to_datetime(batsman['Start Date'])
    
    atitle = name + "'s Moving average (Runs)"
    # Plot the runs in grey colo
    plt.plot(date,runs,"-",color = '0.75')
    
    # Compute and plot moving average
    y_av = movingaverage(runs, 50)
    plt.xlabel('Date')
    plt.ylabel('Runs')
    plt.plot(date, y_av,"b")
    plt.title(atitle)
    plt.text('2002-01-03',150,'Data source-Courtesy:ESPN Cricinfo',
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
    plt.show()
    plt.gcf().clear()
    return


import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 14 Oct 2018
# Function: batsmanPerfBoxHist
# This function makes a box plot showing the mean, median and the 25th & 75th percentile runs. The
# histogram shows the frequency of scoring runs in different run ranges
#
###########################################################################################
# Plot the batting performance as a combined box plot and histogram
def batsmanPerfBoxHist(file, name="A Hitter"):
    '''
    Make a boxplot and a histogram of the runs scored by the batsman
    
    Description
    
    Make a boxplot and histogram of the runs scored by the batsman. Plot the Mean, Median, 25th and 75th quantile
    
    Usage
    
    batsmanPerfBoxHist(file, name="A Hitter")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanDismissals, batsmanMeanStrikeRate, batsmanMovingAverage, batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkar = getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    
    batsman4s("tendulkar.csv","Sachin Tendulkar")
    '''
    batsman = clean(file)
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6   
    
    batsman['Runs']=pd.to_numeric(batsman['Runs'])
    
    plt.subplot(2,1,1)
    sns.boxplot(batsman['Runs'])
    
    plt.subplot(2,1,2);
    atitle = name + "'s" +  " - Runs Frequency vs Runs"
    
    plt.hist(batsman['Runs'],bins=20, edgecolor='black')
    plt.xlabel('Runs')
    plt.ylabel('Strike Rate')
    plt.title(atitle,size=16)
    plt.text(180, 70,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             )
    plt.show()
    plt.gcf().clear()
    return    
    
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 20 Oct 2018
# Function: batsmanPerfForecast
# This function forecasts the batsmans performance based on past performance - 
# To update
###########################################################################################

def batsmanPerfForecast(file, name="A Squarecut"):
    '''
    # To do: Currently ARIMA is used.
    Forecast the batting performance based on past performances using Holt-Winters forecasting
    
    Description
    
    This function forecasts the performance of the batsman based on past performances using HoltWinters forecasting model
    
    Usage
    
    batsmanPerfForecast(file, name="A Squarecut")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanDismissals, batsmanMeanStrikeRate, batsmanMovingAverage, batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkar = getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    batsmanPerfForecast("tendulkar.csv","Sachin Tendulkar")
    
    '''
    
    batsman= clean(file) 
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    runs=batsman['Runs'].astype('float')
    
    # Fit a ARIMA model
    date= pd.to_datetime(batsman['Start Date'])
    df=pd.DataFrame({'date':date,'runs':runs})
    
    df1=df.set_index('date')
    
    model = ARIMA(df1, order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    plt.gcf().clear()
    print(residuals.describe())
    
    
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: batsmanPerfHomeAway
# This plots the batsman's performance in home versus abroad
#
###########################################################################################
def batsmanPerfHomeAway(file,name="A Hitter"):
    '''
    This function analyses the performance of the batsman at home and overseas
    
    Description
    
    This function plots the runs scored by the batsman at home and overseas
    
    Usage
    
    batsmanPerfHomeAway(file, name = "A Hitter")
    Arguments
    
    file	
    CSV file of batsman from ESPN Cricinfo obtained with getPlayerDataSp()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanMovingAverage batsmanRunsPredict batsmanPerfBoxHist bowlerContributionWonLost
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkarSp <-getPlayerDataSp(35320,".","tendulkarsp.csv","batting")
    
    batsmanPerfHomeAway("tendulkarsp.csv","Sachin Tendulkar")
    
    '''
    
    playersp = clean(file)
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    # Create separate DFs for home and away
    home = playersp[playersp['ha'] == 1]
    away = playersp[playersp['ha']==2]
    
    home['venue']="Home"
    away['venue']="Overseas"
    
    df= pd.concat([home,away])
    df['Runs']=  pd.to_numeric(df['Runs'])
    atitle = name + "-" + "- - Runs-Home & overseas" 

    ax = sns.boxplot(x='venue',y='Runs',data=df)
    
    plt.title(atitle)
    plt.text(0.5, 200,'Data source-Courtesy:ESPN Cricinfo',
         horizontalalignment='center',
         verticalalignment='center',
         )
    plt.show()
    plt.gcf().clear()
    return  
       
    
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 30 Jun 2015
# Function: batsmanRunsFreqPerf
# This function computes and plots the Moving Average of the batsman across his career
#
###########################################################################################
# Plot the performance of the batsman as a continous graph
# Create a performance plot between Runs and RunsFrequency 
def batsmanRunsFreqPerf(file, name="A Hookshot"):
    '''
    Calculate and run frequencies in ranges of 10 runs and plot versus Runs the performance of the batsman
    
    Description
    
    This function calculates frequencies of runs in 10 run buckets and plots this percentage
    
    Usage
    
    batsmanRunsFreqPerf(file, name="A Hookshot")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanDismissals, batsmanMovingAverage, batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkar <- getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    
    batsmanRunsFreqPerf("tendulkar.csv","Sachin Tendulkar")
    '''     
    df = clean(file)

    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    runs=pd.to_numeric(df['Runs'])
    
    # Plot histogram
    runs.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    atitle = name + "'s" +  " Runs histogram"
    plt.title(atitle)
    plt.xlabel('Runs')
    plt.grid(axis='y', alpha=0.75)
    plt.text(180, 90,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             )
    plt.show()
    plt.gcf().clear()
    return
    

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 14 Oct 2018
# Function: batsmanRunsLikelihood
# This function used K-Means to compute and plot the runs likelihood for the batsman
# To do - Include scatterplot
###########################################################################################

def batsmanRunsLikelihood(file, name="A Squarecut") :
    '''
    This function uses K-Means to determine the likelihood of the batsman to get runs
    
    Description
    
    This function used K-Means to get the likelihood of getting runs based on clusters of runs the batsman made in the past.It uses K-Means for this.
    
    Usage
    
    batsmanRunsLikelihood(file, name = "A Squarecut")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanMovingAverage batsmanRunsPredict battingPerf3d batsmanContributionWonLost
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    # tendulkar= getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    
    batsmanRunsLikelihood("tendulkar.csv","Sachin Tendulkar")
    ''' 
    batsman =clean(file)
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    data = batsman[['Runs','BF','Mins']]
       
    # Create 3 different clusters
    kmeans = KMeans(n_clusters=3,max_iter=300)
    
    # Compute  the clusters
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    # Get the cluster centroids
    centers = kmeans.cluster_centers_
    centers
    
  # Add a title
    atitle= name + '-' + "Runs Likelihood"
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # Draw vertical line 1st centroid
    x=[centers[0][0],centers[0][0]]
    y=[centers[0][1],centers[0][1]]
    z=[0,centers[0][2]]
    ax.plot(x,y,z,'k-',color='r',alpha=0.8, linewidth=2)
    
    # Draw vertical line 2nd centroid
    x=[centers[1][0],centers[1][0]]
    y=[centers[1][1],centers[1][1]]
    z=[0,centers[1][2]]
    ax.plot(x,y,z,'k-',color='b',alpha=0.8, linewidth=2)
    
    # Draw vertical line 2nd centroid
    x=[centers[2][0],centers[2][0]]
    y=[centers[2][1],centers[2][1]]
    z=[0,centers[2][2]]
    ax.plot(x,y,z,'k-',color='k',alpha=0.8, linewidth=2)
    ax.set_xlabel('BallsFaced')
    ax.set_ylabel('Minutes')
    ax.set_zlabel('Runs');
    plt.title(atitle)
    plt.show()
    plt.gcf().clear()
    return   
    
    
from sklearn.linear_model import LinearRegression
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: batsmanRunsPredict
# This function predicts the runs that will be scored by the batsman for a given numbers
# of balls faced and minutes at crease
#
###########################################################################################
def batsmanRunsPredict(file, newDF, name="A Coverdrive"):
    
    '''
    Predict the runs for the batsman given the Balls Faced and Minutes in crease
    
    Description
    
    Fit a linear regression plane between Runs scored and Minutes in Crease and Balls Faced. This will be used to predict the batsman runs for time in crease and balls faced
    
    Usage
    
    batsmanRunsPredict(file, name="A Coverdrive", newdataframe)
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    newdataframe	
    This is a data frame with 2 columns BF(Balls Faced) and Mins(Minutes)
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    Returns a data frame with the predicted runs for the Balls Faced and Minutes at crease
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanMovingAverage battingPerf3d batsmanContributionWonLost
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    # tendulkar <- getPlayerData(35320,file="tendulkar.csv",type="batting", 
    # homeOrAway=c(1,2), result=c(1,2,4))
    
    # Use a single value for BF and Mins
    BF= 30
    Mins= 20
    
    BF = np.linspace( 10, 400,15)
    Mins =  np.linspace(30,220,15)
    newDF= pd.DataFrame({'BF':BF,'Mins':Mins}
    
    # retrieve the file path of a data file installed with cricketr
    pathToFile <- system.file("data", "tendulkar.csv", package = "cricketr")
    batsmanRunsPredict("tendulkar.csv",newDF, "Sachin Tendulkar")

    '''
    batsman = clean(file)
    df=batsman[['BF','Mins','Runs']]
    df['BF']=pd.to_numeric(df['BF'])
    df['Runs']=pd.to_numeric(df['Runs'])
    xtrain=df.iloc[:,0:2]
    ytrain=df.iloc[:,2]
    linreg = LinearRegression().fit(xtrain, ytrain)
    

    newDF['Runs']=linreg.predict(newDF)
    return(newDF)

import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 13 Oct 2018
# Function: batsmanRunsRanges
# This plots the percentage runs in different run ranges
#
###########################################################################################
def batsmanRunsRanges(file, name= "A Hookshot") :
    '''
    Compute and plot a histogram of the runs scored in ranges of 10
    
    Description
    
    Compute and plot a histogram of the runs scored in ranges of 10
    
    Usage
    
    batsmanRunsRanges(file, name="A Hookshot")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanDismissals, batsmanMovingAverage, batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkar= getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    
    batsmanRunsRanges(pathToFile,"Sachin Tendulkar")
    '''
    
    # Clean file
    batsman = clean(file)
    runs= pd.to_numeric(batsman['Runs'])
    hist, bins = np.histogram(runs, bins = 20)
    midBin=[]
    # Loop through 
    for i in range(1,len(bins)):
        # Find the mean of the bins (Runs)
        midBin.append(np.mean([bins[i-1],bins[i]]))
    
    # Compute binWidth. Subtract '2' to separate the bars
    binWidth=bins[1]-bins[0]-2
    
    # Plot a barplot
    plt.bar(midBin, hist, bins[1]-bins[0]-2, color="blue")
    plt.xlabel('Run ranges')
    plt.ylabel('Frequency')
    
    # Add a title
    atitle= name + '-' + "Runs %  vs Run frequencies"
    plt.title(atitle)
    plt.text(180, 70,'Data source-Courtesy:ESPN Cricinfo',
         horizontalalignment='center',
         verticalalignment='center',
         )
    plt.show()
    plt.gcf().clear()
    return   
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.linear_model import LinearRegression
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 20 Oct 2018
# Function: battingPerf3d
# This function creates a 3D scatter plot of Runs scored vs Balls Faced and Minutes in crease. 
# A regression plane is fitted to this.
#
###########################################################################################

def battingPerf3d(file, name="A Hookshot") :
    '''
    Make a 3D scatter plot of the Runs scored versus the Balls Faced and Minutes at Crease.
    
    Description
    
    Make a 3D plot of the Runs scored by batsman vs Minutes in crease and Balls faced. Fit a linear regression plane
    
    Usage
    
    battingPerf3d(file, name="A Hookshot")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerData()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanDismissals, batsmanMeanStrikeRate, batsmanMovingAverage, batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #  tendulkar<- getPlayerData(35320,file="tendulkar.csv",type="batting", 
    #homeOrAway=[1,2],result=[1,2,4])
    
    battingPerf3d("tendulkar.csv","Sachin Tendulkar")
    '''
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    # Clean the batsman file and create a complete data frame
    batsman = clean(file)
    # Make a 3 D plot and fit a regression plane
    atitle = name +  "- Runs vs BallsFaced & Minutes"
    
    df2=batsman[['BF','Mins','Runs']]
    df2['BF']=pd.to_numeric(df2['BF'])
    df2['Mins']=pd.to_numeric(df2['Mins'])
    df2['Runs']=pd.to_numeric(df2['Runs'])
    
    X=df2.iloc[:,0:2]
    Y=df2.iloc[:,2]

     # Fit a Regression place
    linreg = LinearRegression().fit(X,Y)
    bf= np.linspace(0,400,20)
    mins=np.linspace(0,620,20)
    xx, yy = np.meshgrid(bf,mins)
    xx1=xx.reshape(-1)
    yy1=yy.reshape(-1)
    test=pd.DataFrame({"BallsFaced": xx1, "Minutes":yy1})
    predictedRuns=linreg.predict(test).reshape(20,20)
    
    
    plt3d = plt.figure().gca(projection='3d')
    plt3d.scatter(df2['BF'],df2['Mins'],df2['Runs'])
    plt3d.plot_surface(xx.reshape(20,20),yy,predictedRuns, alpha=0.2)
    plt3d.set_xlabel('BallsFaced')
    plt3d.set_ylabel('Minutes')
    plt3d.set_zlabel('Runs');
    plt.title(atitle)
    plt.show()
    plt.gcf().clear()
    return
    
    
import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerAvgWktsGround
# This function plots the average runs scored by batsman at the ground. The xlabels indicate
# the number of innings at ground
# To do - Append number of matches to Ground
###########################################################################################

def bowlerAvgWktsGround(file, name="A Chinaman"):
    '''
    This function computes and plot the average wickets in different ground

    Description
    
    This function computes the average wickets taken against different grounds by the bowler. It also shows the number innings at each venue
    
    Usage
    
    bowlerAvgWktsGround(file, name = "A Chinaman")
    Arguments
    
    file	
    This is the <bowler>.csv file obtained with an initial getPlayerData()
    name	
    Name of the bowler
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerWktsFreqPercent relativeBowlingER relativeBowlingPerf
    
    Examples
    
    # Get or use the <bowler>.csv obtained with getPlayerData()
    # a <- getPlayerData(30176,file="kumble.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    
    bowlerAvgWktsGround("kumble.csv","Anil Kumble")
    '''

    bowler = cleanBowlerData(file)
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    bowler['Wkts']=pd.to_numeric(bowler['Wkts'])
    
    # Aggregate as sum, mean and count
    df=bowler[['Wkts','Ground']].groupby('Ground').agg(['sum','mean','count'])
    
    #Flatten multi-levels to column names
    df.columns= ['_'.join(col).strip() for col in df.columns.values]
    
    # Reset index
    df1=df.reset_index(inplace=False)
    
    atitle = name + "-" + "'s Average Wickets at Ground"
    
    plt.xticks(rotation="vertical",fontsize=8)
    plt.axhline(y=4, color='r', linestyle=':')
    plt.title(atitle)
    ax=sns.barplot(x='Ground', y="Wkts_mean", data=df1)
    #plt.bar(df1['Ground'],df1['Wkts_mean'])
    plt.text(15, 4,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             )
    plt.show()
    plt.gcf().clear()
    return
    

import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerAvgWktsOpposition
# This function plots the average runs scored by batsman at the ground. The xlabels indicate
# the number of innings at ground
# To do - Append no of matches in Opposition
###########################################################################################

def bowlerAvgWktsOpposition(file, name="A Chinaman"):

    '''
    This function computes and plot the average wickets against different oppositon
    
    Description
    
    This function computes the average wickets taken against different opposition by the bowler. It also shows the number innings against each opposition
    
    Usage
    
    bowlerAvgWktsOpposition(file, name = "A Chinaman")
    Arguments
    
    file	
    This is the <bowler>.csv file obtained with an initial getPlayerData()
    name	
    Name of the bowler
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerWktsFreqPercent relativeBowlingER relativeBowlingPerf bowlerAvgWktsGround
    
    Examples
    
    # Get or use the <bowler>.csv obtained with getPlayerData()
    # kumble <- getPlayerData(30176,file="kumble.csv",type="batting", homeOrAway=c(1,2),result=c(1,2,4))

    bowlerAvgWktsOpposition("kumble.csv","Anil Kumble")
    '''
    bowler = cleanBowlerData(file)  
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    bowler['Wkts']=pd.to_numeric(bowler['Wkts'])
    
    # Aggregate as sum, mean and count
    df=bowler[['Opposition','Wkts']].groupby('Opposition').agg(['sum','mean','count'])
    
    #Flatten multi-levels to column names
    df.columns= ['_'.join(col).strip() for col in df.columns.values]
    
    # Reset index
    df1=df.reset_index(inplace=False)
    
    atitle = name + "-" + "'s Average Wickets vs Opposition"
    
    plt.xticks(rotation="vertical",fontsize=8)
    plt.axhline(y=3, color='r', linestyle=':')
    ax=sns.barplot(x='Opposition', y="Wkts_mean", data=df1)
    plt.title(atitle)
    plt.text(2, 3,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             )
    plt.show()
    plt.gcf().clear()
    return

import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerContributionWonLost
# This plots the bowler's contribution to won and lost matches
#
###########################################################################################
def bowlerContributionWonLost(file,name="A Doosra"):
    '''
    Display the bowler's contribution in matches that were won and those that were lost
    
    Description
    
    Plot the comparative contribution of the bowler in matches that were won and lost as box plots
    
    Usage
    
    bowlerContributionWonLost(file, name = "A Doosra")
    Arguments
    
    file	
    CSV file of bowler from ESPN Cricinfo obtained with getPlayerDataSp()
    name	
    Name of the bowler
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerMovingAverage bowlerPerfForecast checkBowlerInForm
    
    Examples
    
    # Get or use the <bowler>.csv obtained with getPlayerDataSp()
    #kumbleSp <-getPlayerDataSp(30176,".","kumblesp.csv","bowling")

    bowlerContributionWonLost("kumblesp.csv","Anil Kumble")
    '''   
    playersp = cleanBowlerData(file)
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    # Create DFs for won and lost/drawn
    won = playersp[playersp['result'] == 1]
    lost = playersp[(playersp['result']==2) | (playersp['result']==4)]
    won['status']="won"
    lost['status']="lost"
    
    # Stack DFs
    df= pd.concat([won,lost])
    df['Wkts']=  pd.to_numeric(df['Wkts'])
    ax = sns.boxplot(x='status',y='Wkts',data=df)
    atitle = name + "-" + "- Wickets in games won/lost-drawn" 
    plt.xlabel('Status')
    plt.ylabel('Wickets')    
    plt.title(atitle)
    plt.text(0.5, 200,'Data source-Courtesy:ESPN Cricinfo',
         horizontalalignment='center',
         verticalalignment='center',
         )
    plt.show()
    plt.gcf().clear()
    return

import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerCumulativeAvgEconRate
# This function computes and plots the cumulative average economy rate of a bowler
#
###########################################################################################

def bowlerCumulativeAvgEconRate(file,name="A Googly"):
    '''
    Bowler's cumulative average economy rate
    
    Description
    
    This function computes and plots the cumulative average economy rate of a bowler
    
    Usage
    
    bowlerCumulativeAvgEconRate(file,name)
    Arguments
    
    file	
    Data frame
    name	
    Name of batsman
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanCumulativeAverageRuns bowlerCumulativeAvgWickets batsmanCumulativeStrikeRate
    
    Examples
    
    bowlerCumulativeAvgEconRate("kumble.csv","Anil Kumble")
    '''
    bowler=cleanBowlerData(file) 
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    economyRate=pd.to_numeric(bowler['Econ'])
    cumEconomyRate = economyRate.cumsum()/pd.Series(np.arange(1, len(economyRate)+1), economyRate.index)
    
    atitle = name + "- Cumulative Economy Rate vs No of innings"

    plt.xlabel('Innings')
    plt.ylabel('Cumulative Economy Rate')
    plt.title(atitle)
    plt.plot(cumEconomyRate)

    
    plt.text(150,3,'Data source-Courtesy:ESPN Cricinfo',
                 horizontalalignment='center',
                 verticalalignment='center',
                 )  
    
    plt.show()
    plt.gcf().clear()
    return    

import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerCumulativeAvgWickets
# This function computes and plots the cumulative average wickets of a bowler
#
###########################################################################################

def bowlerCumulativeAvgWickets(file,name="A Googly"):
    '''
    Bowler's cumulative average wickets
    
    Description
    
    This function computes and plots the cumulative average wickets of a bowler
    
    Usage
    
    bowlerCumulativeAvgWickets(file,name)
    Arguments
    
    file	
    Data frame
    name	
    Name of batsman
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanCumulativeAverageRuns bowlerCumulativeAvgEconRate batsmanCumulativeStrikeRate
    
    Examples
    
    bowlerCumulativeAvgWickets("kumble.csv","Anil Kumble")
    
    '''
    bowler=cleanBowlerData(file) 
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    wktRate=pd.to_numeric(bowler['Wkts'])
    cumWktRate = wktRate.cumsum()/pd.Series(np.arange(1, len(wktRate)+1), wktRate.index)
    
    atitle = name + "- Cumulative Mean Wicket Rate vs No of innings"

    plt.xlabel('Innings')
    plt.ylabel('Cumulative Mean Wickets')
    plt.title(atitle)
    plt.plot(cumWktRate)

    
    plt.text(150,3,'Data source-Courtesy:ESPN Cricinfo',
                 horizontalalignment='center',
                 verticalalignment='center',
                 )  
    plt.show()
    plt.gcf().clear()
    return  
    

import matplotlib.pyplot as plt
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerEconRate
# This function plots the Frequency percentage of wickets taken for the bowler
#
###########################################################################################
def bowlerEconRate(file, name="A Bowler") :
    '''
    Compute and plot the Mean Economy Rate versus wickets taken
    
    Description
    
    This function computes the mean economy rate for the wickets taken and plot this
    
    Usage
    
    bowlerEconRate(file, name = "A Bowler")
    Arguments
    
    file	
    This is the <bowler>.csv file obtained with an initial getPlayerData()
    name	
    Name of the bowler
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerWktsFreqPercent relativeBowlingER relativeBowlingPerf
    
    Examples
    
    # Get or use the <bowler>.csv obtained with getPlayerData()
    # kumble <- getPlayerData(30176,dir=".", file="kumble.csv",type="batting", 
    # homeOrAway=[1,2],result=[1,2,4])
    
    bowlerEconRate("kumble.csv","Anil Kumble")
    '''

    bowler = cleanBowlerData(file)  
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    bowler['Wkts']=pd.to_numeric(bowler['Wkts']) 
    bowler['Econ']=pd.to_numeric(bowler['Econ']) 
        
    atitle = name + "-" + "- Mean economy rate vs Wkts" 
    df=bowler[['Wkts','Econ']].groupby('Wkts').mean()
    df = df.reset_index(inplace=False)
    ax=plt.plot('Wkts','Econ',data=df)
    plt.xlabel('Wickets')
    plt.ylabel('Economy Rate')
    plt.title(atitle)
    plt.text(6, 3,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             )
    plt.show()
    plt.gcf().clear()
    return 
    


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerMovingAverage
# This function computes and plots the Moving Average of the Wickets taken for a bowler
# across his career
#
###########################################################################################
# Compute a moving average
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def bowlerMovingAverage(file,name="A Doosra") :
    '''
    Compute and plot the moving average of the wickets taken for a bowler
    
    Description
    
    This function plots the wickets taken by a bowler as a time series and plots the moving average over the career
    
    Usage
    
    bowlerMovingAverage(file, name = "A Doosra")
    Arguments
    
    file	
    This is the <bowler>.csv file obtained with an initial getPlayerData()
    name	
    Name of the bowler
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerWktsFreqPercent relativeBowlingER relativeBowlingPerf
    
    Examples
    
    # Get or use the <bowler>.csv obtained with getPlayerData()
    # kumble = getPlayerData(30176,file="kumble.csv",type="bowling", homeOrAway=[1,2],result=[1,2,4])
    
    bowlerMovingAverage("kumble.csv","Anil Kumble")
    '''
    bowler = cleanBowlerData(file)  
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    wkts=pd.to_numeric(bowler['Wkts'])
    date= pd.to_datetime(bowler['Start Date'])
    
    atitle = name + "'s Moving average (Runs)"
    # Plot the runs in grey colo
    plt.plot(date,wkts,"-",color = '0.75')
    y_av = movingaverage(wkts, 50)
    plt.xlabel('Date')
    plt.ylabel('Wickets')
    plt.plot(date, y_av,"b")
    plt.title(atitle)
    plt.text('2002-01-03',150,'Data source-Courtesy:ESPN Cricinfo',
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

    plt.show()
    plt.gcf().clear()
    return 


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 20 Oct 2018
# Function: bowlerPerfForecast
# This function forecasts the bowler's performance based on past performance
#
###########################################################################################
def bowlerPerfForecast(file, name="A Googly"):
    '''
    # To do- Currently based on ARIMA
    Forecast the bowler performance based on past performances using Holt-Winters forecasting
    
    Description
    
    This function forecasts the performance of the bowler based on past performances using HoltWinters forecasting model
    
    Usage
    
    bowlerPerfForecast(file, name = "A Googly")
    Arguments
    
    file	
    This is the <bowler>.csv file obtained with an initial getPlayerData()
    name	
    Name of the bowler
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerEconRate, bowlerMovingAverage, bowlerContributionWonLost
    
    Examples
    
    # Get or use the <bowler>.csv obtained with getPlayerData()
    # kumble = getPlayerData(30176,file="kumble.csv",type="bowling", homeOrAway=[1,2],result=[1,2,4])    
    bowlerPerfForecast("kumble.csv","Anil Kumble")
    '''

    bowler= cleanBowlerData(file) 

    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    wkts=bowler['Wkts'].astype('float')
    
    date= pd.to_datetime(bowler['Start Date'])
    df=pd.DataFrame({'date':date,'Wickets':wkts})
    
    df1=df.set_index('date')
    
    model = ARIMA(df1, order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    atitle=name+"-ARIMA plot"
    plt.title(atitle)
    plt.show()
    
    residuals.plot(kind='kde')
    atitle=name+"-ARIMA plot"
    plt.title(atitle)
    plt.show()
    plt.gcf().clear()
    print(residuals.describe())
    

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerPerfHomeAway
# This plots the bowler's performance home and abroad
#
###########################################################################################
def bowlerPerfHomeAway(file,name="A Googly") :
    '''
    This function analyses the performance of the bowler at home and overseas
    
    Description
    
    This function plots the Wickets taken by the batsman at home and overseas
    
    Usage
    
    bowlerPerfHomeAway(file, name = "A Googly")
    Arguments
    
    file	
    CSV file of the bowler from ESPN Cricinfo (for e.g. Kumble's profile no:30176)
    name	
    Name of bowler
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerMovingAverage bowlerPerfForecast checkBowlerInForm bowlerContributionWonLost
    
    Examples
    
    # Get or use the <bowler>.csv obtained with getPlayerDataSp()
    #kumblesp <-getPlayerDataSp(30176,".","kumblesp.csv","bowling")
    
    bowlerPerfHomeAway(kumblesp.csv,"Anil Kumble")
    '''
    
    playersp = cleanBowlerData(file)
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    #
    home = playersp[playersp['ha'] == 1]
    away = playersp[playersp['ha']==2]
    
    home['venue']="Home"
    away['venue']="Overseas"
    
    df= pd.concat([home,away])
    df['Wkts']=  pd.to_numeric(df['Wkts'])
    atitle = name + "-" + "- - Wickets-Home & overseas" 

    ax = sns.boxplot(x='venue',y='Wkts',data=df)
    plt.xlabel('Venue')
    plt.ylabel('Wickets')    
    plt.title(atitle)
    plt.text(0.5, 200,'Data source-Courtesy:ESPN Cricinfo',
         horizontalalignment='center',
         verticalalignment='center',
         )
    plt.show()
    plt.gcf().clear()
    return 
       
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerWktsFreqPercent
# This function plots the Frequency percentage of wickets taken for the bowler
#
###########################################################################################
def bowlerWktsFreqPercent(file, name="A Bowler"):
    '''
    Plot the Wickets Frequency as a percentage against wickets taken
    
    Description
    
    This function calculates the Wickets frequency as a percentage of total wickets taken and plots this agains the wickets taken.
    
    Usage
    
    bowlerWktsFreqPercent(file, name="A Bowler")
    Arguments
    
    file	
    This is the <bowler>.csv file obtained with an initial getPlayerData()
    name	
    Name of the bowler
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerWktsFreqPercent relativeBowlingER relativeBowlingPerf
    
    Examples
    
    # Get or use the <bowler>.csv obtained with getPlayerData()
    # a =getPlayerData(30176,file="kumble.csv",type="bowling", homeOrAway=[1,2],result=[1,2,4])
    
    bowlerWktsFreqPercent("kumble.csv","Anil Kumble")
    '''
    
    bowler = cleanBowlerData(file)
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    # Create a table of wickets
    wkts = pd.to_numeric(bowler['Wkts'])
    
    
    wkts.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    atitle = name + "'s" +  " Wickets histogram"
    plt.title(atitle)
    plt.xlabel('Wickets')
    plt.grid(axis='y', alpha=0.75)
    plt.text(5,10,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             )
    plt.show()
    plt.gcf().clear()
    return
    

import seaborn as sns
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: bowlerWktsRunsPlot
# This function makes boxplot of Wickets versus Runs concded
###########################################################################################
def bowlerWktsRunsPlot(file, name="A Googly"):
    '''
    Compute and plot the runs conceded versus the wickets taken
    
    Description
    
    This function creates boxplots on the runs conceded for wickets taken for the bowler
    
    Usage
    
    bowlerWktsRunsPlot(file, name = "A Googly")
    Arguments
    
    file	
    This is the <bowler>.csv file obtained with an initial getPlayerData()
    name	
    Name of the bowler
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerWktsFreqPercent relativeBowlingER relativeBowlingPerf bowlerHistWickets
    
    Examples
    
    # Get or use the <bowler>.csv obtained with getPlayerData()
    # kumble =getPlayerData(30176,file="kumble.csv",type="bowling", homeOrAway=[1,2],result=[1,2,4])
    
    bowlerWktsRunsPlot("kumble.csv","Anil Kumble")
    '''
    bowler = cleanBowlerData(file)
    # Set figure size
    rcParams['figure.figsize'] = 10,6  

    atitle = name + "- Wickets vs Runs conceded"
    ax = sns.boxplot(x='Wkts', y='Runs', data=bowler)
    plt.title(atitle)
    plt.xlabel('Wickets')
    plt.show()
    plt.gcf().clear()
    return
    
import pandas as pd
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 11 Oct 2018
# Function : clean
# This function cleans the batsman's data file and returns the cleaned data frame for use in
# other functions
##########################################################################################
def clean(batsmanCSV):
    '''
    Create a batsman data frame given the batsman's CSV file
    
    Description
    
    The function removes rows from the batsman dataframe where the batsman did not bat (DNB) or the team did not bat (TDNB). COnverts not outs '*' (97*, 128*) to 97,128 by stripping the '*' character. It picks all the complete cases and returns the data frame
    
    Usage
    
    clean(file)
    Arguments
    
    file	
    CSV file with the batsman data obtained with getPlayerData
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    Returns the cleaned batsman dataframe
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html https://gigadom.wordpress.com/
    
    See Also
    
    cleanBowlerData getPlayerData batsman4s batsmanMovingAverage
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerData()
    #tendulkar = getPlayerData(35320,file="tendulkar.csv",type="batting", homeOrAway=[1,2],result=[1,2,4])
    
    clean(pathToFile)
    '''
    
    df = pd.read_csv(batsmanCSV,na_values=['-'])
      
    a = df['Runs'] != "DNB"
    batsman = df[a]
      
    # Remove rows with 'TDNB'
    c =batsman['Runs'] != "TDNB"
    batsman = batsman[c]
      
    # Remove rows with absent
    d = batsman['Runs'] != "absent"
    batsman = batsman[d]
      
    # Remove the "* indicating not out
    batsman['Runs']= batsman['Runs'].str.replace(r"[*]","")
      
    # Drop rows which have NA
    batsman = batsman.dropna()
      
      
    #Return the data frame 
    return(batsman)
  
  
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function : cleanBowlerData
# This function cleans the bowler's data file and returns the cleaned data frame for use in
# other functions
##########################################################################################
def cleanBowlerData(file):
    '''
    Clean the bowlers data frame
    
    Description
    
    Clean the bowler's CSV fileand remove rows DNB(Did not bowl) & TDNB (Team did not bowl). Also normalize all 8 ball over to a 6 ball over for earlier bowlers
    
    Usage
    
    cleanBowlerData(file)
    Arguments
    
    file	
    The <bowler>.csv file
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    A cleaned bowler data frame with complete cases
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    clean
    
    Examples
    
    
    # Get bowling data and store in file for future
    # kumble <- getPlayerData(30176,dir="./mytest", file="kumble.csv",type="bowling", 
    # homeOrAway=[1],result=[1,2])
    cleanBowlerData(pathToFile)
    '''
    # Read the <bowler>.csv file
    df = pd.read_csv(file,na_values=['-'])
    
    # Remove rows with did not bowl
    
    a = df['Overs']!= "DNB"
    df = df[a]
    
    # Remove rows with 'TDNB' - team did not bowl
    c =df['Overs'] != "TDNB"
    df = df[c]
    
    # Get all complete cases
    bowlerComplete = df.dropna(axis=1)
    # Normalize overs which had 8 balls per over to the number of overs if there 8 balls per over
    if bowlerComplete.columns[2] =="BPO":
        bowlerComplete['Overs'] = pd.to_numeric(bowlerComplete['Overs']) *8/6
    return(bowlerComplete)
    
    
import pandas as pd
import os
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 11 Oct 2018
# Function : getPlayerData
# This function gets the data of batsman/bowler and returns the data frame. This data frame can
# stored for use in other functions
##########################################################################################
def getPlayerData(profile,opposition="",host="",dir="./data",file="player001.csv",type="batting",
                         homeOrAway=[1,2],result=[1,2,4],create=True) :
    '''
    Get the player data from ESPN Cricinfo based on specific inputs and store in a file in a given directory
    
    Description
    
    Get the player data given the profile of the batsman. The allowed inputs are home,away or both and won,lost or draw of matches. The data is stored in a <player>.csv file in a directory specified. This function also returns a data frame of the player
    
    Usage
    
    getPlayerData(profile,opposition="",host="",dir="./data",file="player001.csv",
    type="batting", homeOrAway=c(1,2),result=c(1,2,4))
    Arguments
    
    profile	
    This is the profile number of the player to get data. This can be obtained from http://www.espncricinfo.com/ci/content/player/index.html. Type the name of the player and click search. This will display the details of the player. Make a note of the profile ID. For e.g For Sachin Tendulkar this turns out to be http://www.espncricinfo.com/india/content/player/35320.html. Hence the profile for Sachin is 35320
    opposition	
    The numerical value of the opposition country e.g.Australia,India, England etc. The values are Australia:2,Bangladesh:25,England:1,India:6,New Zealand:5,Pakistan:7,South Africa:3,Sri Lanka:8, West Indies:4, Zimbabwe:9
    host	
    The numerical value of the host country e.g.Australia,India, England etc. The values are Australia:2,Bangladesh:25,England:1,India:6,New Zealand:5,Pakistan:7,South Africa:3,Sri Lanka:8, West Indies:4, Zimbabwe:9
    dir	
    Name of the directory to store the player data into. If not specified the data is stored in a default directory "./data". Default="./data"
    file	
    Name of the file to store the data into for e.g. tendulkar.csv. This can be used for subsequent functions. Default="player001.csv"
    type	
    type of data required. This can be "batting" or "bowling"
    homeOrAway	
    This is a list with either 1,2 or both. 1 is for home 2 is for away
    result	
    This is a list that can take values 1,2,4. 1 - won match 2- lost match 4- draw
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    Returns the player's dataframe
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    getPlayerDataSp
    
    Examples
    
    ## Not run: 
    # Both home and away. Result = won,lost and drawn
    tendulkar = getPlayerData(35320,dir=".", file="tendulkar1.csv",
    type="batting", homeOrAway=[1,2],result=[1,2,4])
    
    # Only away. Get data only for won and lost innings
    tendulkar = getPlayerData(35320,dir=".", file="tendulkar2.csv",
    type="batting",homeOrAway=[2],result=[1,2])
    
    # Get bowling data and store in file for future
    kumble = getPlayerData(30176,dir=".",file="kumble1.csv",
    type="bowling",homeOrAway=[1],result=[1,2])
    
    #Get the Tendulkar's Performance against Australia in Australia
    tendulkar = getPlayerData(35320, opposition = 2,host=2,dir=".", 
    file="tendulkarVsAusInAus.csv",type="batting") 
    
    '''

    # Initial url to ""
    url =""
    suburl1 = "http://stats.espncricinfo.com/ci/engine/player/"
    suburl2 ="?class=1;"
    suburl3 = "template=results;"
    suburl4 = "view=innings"
    
    #Set opposition
    theOpposition = "opposition=" + opposition + ";"
    
    # Set host country
    hostCountry = "host=" + host + ";"
    
    # Create a profile.html with the profile number
    player = str(profile) + ".html"
       
    
    # Set the home or away
    str1=str2=""
    #print(len(homeOrAway))
    for i  in homeOrAway:
        if i == 1:
             str1 = str1 + "home_or_away=1;"
        elif i == 2:
             str1 = str1 + "home_or_away=2;"
    HA= str1
    
    # Set the type batting or bowling
    t = "type=" + type + ";"
    
    # Set the result based on input
    str2=""
    for i in result:    
        if i == 1:
            str2 = str2+ "result=1;"        
        elif i == 2:
            str2 = str2 + "result=2;"          
        elif i == 4:
            str2 = str2 + "result=4;"
    
    result =  str2 
    
    # Create composite URL
    url = suburl1 + player + suburl2 + hostCountry + theOpposition + HA + result + suburl3 + t + suburl4
    #print(url)
    # Read the data from ESPN Cricinfo
    dfList= pd.read_html(url)
    
    # Choose appropriate table from list of returned tables
    df=dfList[3]
    colnames= df.columns
    # Select coiumns based on batting or bowling
    if type=="batting" : 
        # Select columns [1:9,11,12,13]
        cols = list(range(0,9))
        cols.extend([10,11,12])
    elif type=="bowling":
        # Check if there are the older version of 8 balls per over (BPO) column
        # [1:8,10,11,12]
        
        # Select BPO column for older bowlers
        if colnames[1] =="BPO":
            # [1:8,10,11,12]
             cols = list(range(0,9))
             cols.extend([10,11,12])
        else:
            # Select columns [1:7,9,10,11]
             cols = list(range(0,8))
             cols.extend([8,9,10])
    
    
    #Subset the necessary columns
    df1 = df.iloc[:, cols]
    
    if not os.path.exists(dir):
        os.mkdir(dir)
        #print("Directory " , dir ,  " Created ")
    else:    
        pass
        #print("Directory " , dir ,  " already exists, writing to this folder")
    
    # Create path
    path= os.path.join(dir,file)
    
    if create:
        # Write to file 
        df1.to_csv(path)

    # Return the data frame
    return(df1)
    
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: getPlayerDataSp
# This function is a specialized version of getPlayer Data. This function gets the players data 
# along with details on matches' venue( home/abroad) and the result (won,lost,drawn) as 
# 2 separate columns
#
###########################################################################################
def getPlayerDataSp(profileNo,tdir="./data",tfile="player001.csv",ttype="batting"):
    '''
    Get the player data along with venue and result status
    
    Description
    
    This function is a specialized version of getPlayer Data. This function gets the players data along with details on matches' venue (home/abroad) and the result of match(won,lost,drawn) as 2 separate columns (ha & result). The column ha has 1:home and 2: overseas. The column result has values 1:won , 2;lost and :drawn match
    
    Usage
    
    getPlayerDataSp(profileNo, tdir = "./data", tfile = "player001.csv", 
    ttype = "batting")
    Arguments
    
    profileNo	
    This is the profile number of the player to get data. This can be obtained from http://www.espncricinfo.com/ci/content/player/index.html. Type the name of the player and click search. This will display the details of the player. Make a note of the profile ID. For e.g For Sachin Tendulkar this turns out to be http://www.espncricinfo.com/india/content/player/35320.html. Hence the profile for Sachin is 35320
    tdir	
    Name of the directory to store the player data into. If not specified the data is stored in a default directory "./data". Default="./tdata"
    tfile	
    Name of the file to store the data into for e.g. tendulkar.csv. This can be used for subsequent functions. Default="player001.csv"
    ttype	
    type of data required. This can be "batting" or "bowling"
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    Returns the player's dataframe along with the homeAway and the result columns
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    getPlayerData
    
    Examples
    
    ## Not run: 
    # Only away. Get data only for won and lost innings
    tendulkar = getPlayerDataSp(35320,tdir="..", tfile="tendulkarsp.csv",ttype="batting")
    
    # Get bowling data and store in file for future
    kumble = getPlayerDataSp(30176,tdir="..",tfile="kumblesp.csv",ttype="bowling")
    
    ## End(Not run)
    '''

    # Get the data for the player i
    # Home & won
    hw = getPlayerData(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[1],result=[1],type=ttype,create=False)
    # Home & lost
    hl = getPlayerData(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[1],result=[2],type=ttype,create=False)
    # Home & drawn
    hd = getPlayerData(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[1],result=[4],type=ttype,create=False)
    # Away and won
    aw = getPlayerData(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[2],result=[1],type=ttype,create=False)
    #Away and lost
    al = getPlayerData(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[2],result=[2],type=ttype,create=False)
    # Away and drawn
    ad = getPlayerData(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[2],result=[4],type=ttype,create=False)
    
    # Set the values as follows
    # ha := home = 1, away =2
    # result= won = 1, lost = 2, drawn=4
    hw['ha'] = 1
    hw['result'] = 1
    
    hl['ha'] = 1
    hl['result'] = 2
    
    hd['ha'] = 1
    hd['result'] = 4
    
    aw['ha'] = 2
    aw['result'] = 1
    
    al['ha'] = 2
    al['result'] = 2
    
    ad['ha'] =  2
    ad['result'] =  4
      
    if not os.path.exists(tdir):
        os.mkdir(dir)
        #print("Directory " , dir ,  " Created ")
    else:    
        pass
        #print("Directory " , dir ,  " already exists, writing to this folder")
    
    # Create path
    path= os.path.join(tdir,tfile)
        
    df= pd.concat([hw,hl,hd,aw,al,ad])
    
    # Write to file 
    df.to_csv(path,index=False)
    
    return(df)
    
import pandas as pd
import os
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 7 Oct 2018
# Function : getPlayerDataOD
# This function gets the One Day data of batsman/bowler and returns the data frame. This data frame can
# stored for use in other functions
##########################################################################################
def getPlayerDataOD(profile,opposition="",host="",dir="./data",file="player001.csv",type="batting",
                         homeOrAway=[1,2,3],result=[1,2,3,5],create=True) :
    '''
    Get the One day player data from ESPN Cricinfo based on specific inputs and store in a file in a given directory
    
    Description
    
    Get the player data given the profile of the batsman. The allowed inputs are home,away or both and won,lost or draw of matches. The data is stored in a <player>.csv file in a directory specified. This function also returns a data frame of the player
    
    Usage
    
    getPlayerDataOD(profile, opposition="",host="",dir = "../", file = "player001.csv", 
    type = "batting", homeOrAway = c(1, 2, 3), result = c(1, 2, 3,5))
    Arguments
    
    profile	
    This is the profile number of the player to get data. This can be obtained from http://www.espncricinfo.com/ci/content/player/index.html. Type the name of the player and click search. This will display the details of the player. Make a note of the profile ID. For e.g For Virender Sehwag this turns out to be http://www.espncricinfo.com/india/content/player/35263.html. Hence the profile for Sehwag is 35263
    opposition	    The numerical value of the opposition country e.g.Australia,India, England etc. The values are Australia:2,Bangladesh:25,Bermuda:12, England:1,Hong Kong:19,India:6,Ireland:29, Netherlands:15,New Zealand:5,Pakistan:7,Scotland:30,South Africa:3,Sri Lanka:8,United Arab Emirates:27, West Indies:4, Zimbabwe:9; Africa XI:405 Note: If no value is entered for opposition then all teams are considered
    host	    The numerical value of the host country e.g.Australia,India, England etc. The values are Australia:2,Bangladesh:25,England:1,India:6,Ireland:29,Malaysia:16,New Zealand:5,Pakistan:7, Scotland:30,South Africa:3,Sri Lanka:8,United Arab Emirates:27,West Indies:4, Zimbabwe:9 Note: If no value is entered for host then all host countries are considered
    dir	
    Name of the directory to store the player data into. If not specified the data is stored in a default directory "../data". Default="../data"
    file	
    Name of the file to store the data into for e.g. tendulkar.csv. This can be used for subsequent functions. Default="player001.csv"
    type	
    type of data required. This can be "batting" or "bowling"
    homeOrAway	
    This is vector with either or all 1,2, 3. 1 is for home 2 is for away, 3 is for neutral venue
    result	
    This is a vector that can take values 1,2,3,5. 1 - won match 2- lost match 3-tied 5- no result
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    Returns the player's dataframe
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    getPlayerDataSp getPlayerData
    
    Examples
    
    
    ## Not run: 
    # Both home and away. Result = won,lost and drawn
    sehwag =getPlayerDataOD(35263,dir="../cricketr/data", file="sehwag1.csv",
    type="batting", homeOrAway=[1,2],result=[1,2,3,4])
    
    # Only away. Get data only for won and lost innings
    sehwag = getPlayerDataOD(35263,dir="../cricketr/data", file="sehwag2.csv",
    type="batting",homeOrAway=[2],result=[1,2])
    
    # Get bowling data and store in file for future
    malinga = getPlayerData(49758,dir="../cricketr/data",file="malinga1.csv",
    type="bowling")
    
    # Get Dhoni's ODI record in Australia against Australua
    dhoni = getPlayerDataOD(28081,opposition = 2,host=2,dir=".",
    file="dhoniVsAusinAusOD",type="batting")
    
    ## End(Not run)
    '''

    # Initial url to ""
    url =""
    suburl1 = "http://stats.espncricinfo.com/ci/engine/player/"
    suburl2 ="?class=2;"
    suburl3 = "template=results;"
    suburl4 = "view=innings"
    
    #Set opposition
    theOpposition = "opposition=" + opposition + ";"
    
    # Set host country
    hostCountry = "host=" + host + ";"
    
    # Create a profile.html with the profile number
    player = str(profile) + ".html"
       
    
    # Set the home or away
    str1=str2=""
    #print(len(homeOrAway))
    for i  in homeOrAway:
        if i == 1:
             str1 = str1 + "home_or_away=1;"
        elif i == 2:
             str1 = str1 + "home_or_away=2;"
        elif i == 3:
             str1 = str1 + "home_or_away=3;"
    HA= str1
    
    # Set the type batting or bowling
    t = "type=" + type + ";"
    
    # Set the result based on input
    str2=""
    for i in result:    
        if i == 1:
            str2 = str2+ "result=1;"        
        elif i == 2:
            str2 = str2 + "result=2;"          
        elif i == 3:
            str2 = str2 + "result=3;"
        elif i == 5:
            str2 = str2 + "result=5;"
    
    result =  str2 
    
    # Create composite URL
    url = suburl1 + player + suburl2 + hostCountry + theOpposition + HA + result + suburl3 + t + suburl4
    #print(url)
    # Read the data from ESPN Cricinfo
    dfList= pd.read_html(url)
    
    # Choose appropriate table from list of returned tables
    df=dfList[3]
    colnames= df.columns
    # Select coiumns based on batting or bowling
    if type=="batting" : 
        # Select columns [1:9,11,12,13]
        cols = list(range(0,9))
        cols.extend([10,11,12])
    elif type=="bowling":
        # Check if there are the older version of 8 balls per over (BPO) column
        # [1:8,10,11,12]
        
        # Select BPO column for older bowlers
        if colnames[1] =="BPO":
            # [1:8,10,11,12]
             cols = list(range(0,9))
             cols.extend([10,11,12])
        else:
            # Select columns [1:7,9,10,11]
             cols = list(range(0,8))
             cols.extend([8,9,10])
    
    
    #Subset the necessary columns
    df1 = df.iloc[:, cols]
    
    if not os.path.exists(dir):
        os.mkdir(dir)
        #print("Directory " , dir ,  " Created ")
    else:    
        pass
        #print("Directory " , dir ,  " already exists, writing to this folder")
    
    # Create path
    path= os.path.join(dir,file)
    
    if create:
        # Write to file 
        df1.to_csv(path)

    # Return the data frame
    return(df1)
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: relativeBatsmanCumulativeAvgRuns
# This function computes and plots the relative cumulative average runs of batsmen
#
###########################################################################################
def relativeBatsmanCumulativeAvgRuns(filelist, names):
    '''
    Relative batsman's cumulative average runs
    
    Description
    
    This function computes and plots the relative cumulative average runs of batsmen
    
    Usage
    
    relativeBatsmanCumulativeAvgRuns(frames, names)
    Arguments
    
    frames	
    This is a list of <batsman>.csv files obtained with an initial getPlayerData()
    names	
    A list of batsmen names who need to be compared
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    relativeBatsmanCumulativeStrikeRate relativeBowlerCumulativeAvgEconRate relativeBowlerCumulativeAvgWickets
    
    Examples
    


    batsmen=["tendulkar.csv","dravid.csv","ganguly.csv"]
    names = ["Tendulkar","Dravid","Ganguly"]
    relativeBatsmanCumulativeAvgRuns(batsmen,names)

    '''
    df1=pd.DataFrame()
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    for idx,file in enumerate(filelist):
        df=clean(file)
        runs=pd.to_numeric(df['Runs'])
        df1[names[idx]] = runs.cumsum()/pd.Series(np.arange(1, len(runs)+1), runs.index)

    df1.plot()
    plt.xlabel('Innings')
    plt.ylabel('Cumulative Average Runs')
    plt.title('Relative batsmen cumulative average runs')
    plt.text(180, 50,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             ) 
    plt.show()
    plt.gcf().clear()
    return
        
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: relativeBatsmanCumulativeAvgRuns
# This function computes and plots the relative cumulative average runs of batsmen
#
###########################################################################################
def relativeBatsmanCumulativeStrikeRate (filelist, names):
    '''
    Relative batsmen cumulative average strike rate
    
    Description
    
    This function computes and plots the cumulative average strike rate of batsmen
    
    Usage
    
    relativeBatsmanCumulativeStrikeRate(frames, names)
    Arguments
    
    frames	
    This is a list of <batsman>.csv files obtained with an initial getPlayerData()
    names	
    A list of batsmen names who need to be compared
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    relativeBatsmanCumulativeAvgRuns relativeBowlerCumulativeAvgEconRate relativeBowlerCumulativeAvgWickets
    
    Examples

    batsmen=["tendulkar.csv","dravid.csv","ganguly.csv"]
    names = ["Tendulkar","Dravid","Ganguly"]
    relativeBatsmanCumulativeAvgRuns(batsmen,names)

    '''
    df1=pd.DataFrame()
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    for idx,file in enumerate(filelist):
        df=clean(file)
        strikeRate=pd.to_numeric(df['SR'])
        df1[names[idx]] = strikeRate.cumsum()/pd.Series(np.arange(1, len(strikeRate)+1), strikeRate.index)
          
    df1.plot()
    plt.xlabel('Innings')
    plt.ylabel('Cumulative Strike Rate')
    plt.title('Relative batsmen cumulative strike rate')
    plt.text(180, 50,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             ) 
    plt.show()
    plt.gcf().clear()
    return 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: relativeBowlerCumulativeAvgEconRate
# This function computes and plots the relativecumulative average economy rates bowlers
#
###########################################################################################

def relativeBowlerCumulativeAvgEconRate(filelist, names):
    '''
    Relative Bowler's cumulative average economy rate
    
    Description
    
    This function computes and plots the relative cumulative average economy rate of bowlers
    
    Usage
    
    relativeBowlerCumulativeAvgEconRate(frames, names)
    Arguments
    
    frames	
    This is a list of <bowler>.csv files obtained with an initial getPlayerData()
    names	
    A list of Twenty20 bowlers names who need to be compared
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    relativeBatsmanCumulativeAvgRuns relativeBowlerCumulativeAvgWickets relativeBatsmanCumulativeStrikeRate
    
    Examples
      
    frames = ["kumble.csv","warne.csv","murali.csv"]
    names = ["Kumble","Warne","Murali"]
    relativeBowlerCumulativeAvgEconRate(frames,names)
    
    '''
    
    df1=pd.DataFrame()
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    for idx,file in enumerate(filelist):
        #print(idx)
        #print(file)
        bowler = cleanBowlerData(file)  
        economyRate=pd.to_numeric(bowler['Econ'])
        df1[names[idx]]= economyRate.cumsum()/pd.Series(np.arange(1, len(economyRate)+1), economyRate.index)
        
          
    df1.plot()    
    plt.xlabel('Innings')
    plt.ylabel('Cumulative Average Econmy Rate')
    plt.title('Relative Cumulative Average Economy Rate')
    plt.text(150, 3,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             ) 
    plt.show()
    plt.gcf().clear()
    return
        
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date :  19 Oct 2018
# Function: relativeBowlerCumulativeAvgWickets
# This function computes and plots the relative cumulative average wickets of bowlers
#
###########################################################################################

def relativeBowlerCumulativeAvgWickets(filelist, names):
    '''
    Relative bowlers cumulative average wickets
    
    Description
    
    This function computes and plots the relative cumulative average wickets of a bowler
    
    Usage
    
    relativeBowlerCumulativeAvgWickets(frames, names)
    Arguments
    
    frames	
    This is a list of <bowler>.csv files obtained with an initial getPlayerData()
    names	
    A list of Twenty20 bowlers names who need to be compared
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    relativeBatsmanCumulativeAvgRuns relativeBowlerCumulativeAvgEconRate relativeBatsmanCumulativeStrikeRate
    
    Examples
    
    ## Not run: )
    
    # Retrieve the file path of a data file installed with cricketr

    
    frames = ["kumble.csv","warne.csv","murali.csv"]
    names = ["Kumble","Warne","Murali"]
    relativeBowlerCumulativeAvgEconRate(frames,names)
    '''
    
    df1=pd.DataFrame()
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    for idx,file in enumerate(filelist):
        bowler = cleanBowlerData(file)  
        wkts=pd.to_numeric(bowler['Wkts']) 
        df1[names[idx]]= wkts.cumsum()/pd.Series(np.arange(1, len(wkts)+1), wkts.index)
        
          
    df1.plot()    
    plt.xlabel('Innings')
    plt.ylabel('Cumulative Average Wicket Rate')
    plt.title('Relative Cumulative Average Wicket Rate')
    plt.text(150, 3,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             ) 
    plt.show()
    plt.gcf().clear()
    return

    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 19 Oct 2018
# Function: relativeBowlingER
# This function computes and plots the relative bowling Economy Rate of the bowlers
#
###########################################################################################


def relativeBowlingER(filelist, names):

    df1=pd.DataFrame()
    # Set figure size
    rcParams['figure.figsize'] = 10,6  
    
    for idx,file in enumerate(filelist):
        bowler = cleanBowlerData(file)  
        bowler['Wkts']=pd.to_numeric(bowler['Wkts']) 
        bowler['Econ']=pd.to_numeric(bowler['Econ']) 
        df=bowler[['Wkts','Econ']].groupby('Wkts').mean()
        df1[names[idx]]=bowler[['Wkts','Econ']].groupby('Wkts').mean()
        
          
    df1.plot()    
    plt.xlabel('Wickets')
    plt.ylabel('Economy Rate')
    plt.title("Relative Bowling Economy Rate vs Wickets")
    plt.text(5, 3,'Data source-Courtesy:ESPN Cricinfo',
             horizontalalignment='center',
             verticalalignment='center',
             ) 
    plt.show()
    plt.gcf().clear()
    return


##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 10 Oct 2018
# Function: batsmanScoringRateODTT
# This function computes and plots the batsman scoring rate of a One Day batsman
# or a Twenty20 batsman
#

###########################################################################################
def batsmanScoringRateODTT(file, name="A Hookshot"):
    '''
    Compute and plot the predicted scoring rate for a One day batsman or Twenty20
    
    Description
    
    This function computes and plots a 2nd order polynomial between the balls faced and runs scored for ODI or Twenty20
    
    Usage
    
    batsmanScoringRateODTT(file, name = "A Hookshot")
    Arguments
    
    file	
    This is the <batsman>.csv file obtained with an initial getPlayerDataOD() or getPlayerTT()
    name	
    Name of the batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsman6s relativeBatsmanSRODTT relativeRunsFreqPerfODTT
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerDataOD() or or getPlayerTT()
    #sehwag =-getPlayerDataOD(35263,dir="./mytest", file="sehwag.csv",type="batting",
    # homeOrAway=c(1,2,3),result=c(1,2,3,5))
    
    # Retrieve the file path of a data file installed with cricketr
    batsmanScoringRateODTT("sehwag.csv","Sehwag")
    
    '''   
    # Clean the batsman file and create a complete data frame

    df = clean(file)
    df['BF'] = pd.to_numeric(df['BF'])
    df['Runs'] = pd.to_numeric(df['Runs'])
    
    df1 = df[['BF','Runs']].sort_values(by=['BF'])
    
    
    # Set figure size
    rcParams['figure.figsize'] = 10,6
    
    # Get numnber of 4s and runs scored
    bf = pd.to_numeric(df1['BF'])
    runs = pd.to_numeric(df1['Runs'])
    
     
    atitle = name + "-" + "Balls Faced vs Runs scored" 
    
    # Plot no of 4s and a 2nd order curve fit   
    plt.scatter(bf,runs, alpha=0.5)
    plt.xlabel('Balls Faced')
    plt.ylabel('Runs')
    plt.title(atitle)
    
    # Create a polynomial of degree 2
    poly = PolynomialFeatures(degree=2)
    bfPoly = poly.fit_transform(bf.reshape(-1,1))
    linreg = LinearRegression().fit(bfPoly,runs)
    
    plt.plot(bf,linreg.predict(bfPoly),'-r')

    
        # Predict the number of runs for 50 balls faced
    b=poly.fit_transform((np.array(50)))
    c=linreg.predict(b)
    plt.axhline(y=c, color='b', linestyle=':')
    plt.axvline(x=50, color='b', linestyle=':')
    
    
    # Predict the number of runs for 100 balls faced
    b=poly.fit_transform((np.array(100)))
    c=linreg.predict(b)
    plt.axhline(y=c, color='b', linestyle=':')
    plt.axvline(x=100, color='b', linestyle=':')
    
    plt.text(180, 0.5,'Data source-Courtesy:ESPN Cricinfo',
         horizontalalignment='center',
         verticalalignment='center',
         )
    plt.show()
    plt.gcf().clear()
    return


##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 10 Nov 2018
# Function: batsman4s6s
# This function computes and plots the percent of 4s,6s in total runs
#
###########################################################################################

def batsman4s6s(frames, names) :
    '''
    Compute and plot a stacked barplot of runs,4s and 6s
    
    Description
    
    Compute and plot a stacked barplot of percentages of runs in (1s,2s and 3s),4s and 6s
    
    Usage
    
    batsman4s6s(frames, names)
    Arguments
    
    frames	
    List of batsman
    names	
    Names of batsman
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    batsmanScoringRateODTT, relativeRunsFreqPerfODTT, batsmanPerfBoxHist
    
    Examples
    
    # Get or use the <batsman>.csv obtained with getPlayerDataOD()
    frames = ["./sehwag.csv","./devilliers.csv","./gayle.csv"]
    names = ["Sehwag","De Villiurs","Gayle"]
    
    batsman4s6s(frames,names)
    
    
    '''

    df2=pd.DataFrame()
    for file in frames:
        df = clean(file)
        runs = pd.to_numeric(df['Runs']).sum()
        x4s = (pd.to_numeric(df['4s']) * 4).sum()
        x6s = (pd.to_numeric(df['6s']) * 6).sum()
        # Find numbers of runs from 1,2 and 3s
        
        runs = runs - (x4s +x6s)
        a=[runs,x4s,x6s]
        df1= pd.DataFrame(a)
        df2=pd.concat([df2,df1],axis=1)
    
    
    df2.columns=names
    df3=df2.T
    df3.columns=['Runs','4s','6s']
    df3.plot(kind="bar",stacked=True)
    plt.show()
    plt.gcf().clear()
    return

##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 11 Nov 2018
# Function: getPlayerDataSpOD
# This function is a specialized version of getPlayerDataOD. This function gets the players data 
# along with details on matches' venue( home/away/neutral) and the result (won,lost,drawn,tie) as 
# 2 separate columns
#
###########################################################################################
def getPlayerDataSpOD(profileNo,tdir="./data",tfile="player001.csv",ttype="batting"):
    '''
    Get the player data along with venue and result status
    
    Description
    
    This function is a specialized version of getPlayer Data. This function gets the players data along with details on matches' venue (home/abroad) and the result of match(won,lost,drawn) as 2 separate columns (ha & result). The column ha has 1:home and 2: overseas. The column result has values 1:won , 2;lost and :drawn match
    
    Usage
    
    getPlayerDataSp(profileNo, tdir = "./data", tfile = "player001.csv", 
    ttype = "batting")
    Arguments
    
    profileNo	
    This is the profile number of the player to get data. This can be obtained from http://www.espncricinfo.com/ci/content/player/index.html. Type the name of the player and click search. This will display the details of the player. Make a note of the profile ID. For e.g For Sachin Tendulkar this turns out to be http://www.espncricinfo.com/india/content/player/35320.html. Hence the profile for Sachin is 35320
    tdir	
    Name of the directory to store the player data into. If not specified the data is stored in a default directory "./data". Default="./tdata"
    tfile	
    Name of the file to store the data into for e.g. tendulkar.csv. This can be used for subsequent functions. Default="player001.csv"
    ttype	
    type of data required. This can be "batting" or "bowling"
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    Returns the player's dataframe along with the homeAway and the result columns
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    getPlayerData
    
    Examples
    
    ## Not run: 
    # Only away. Get data only for won and lost innings
    tendulkar = getPlayerDataSp(35320,tdir="..", tfile="tendulkarsp.csv",ttype="batting")
    
    # Get bowling data and store in file for future
    kumble = getPlayerDataSp(30176,tdir="..",tfile="kumblesp.csv",ttype="bowling")
    
    ## End(Not run)
    '''

    # Get the data for the player i
    # Home & won
    hw = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[1],result=[1],type=ttype,create=False)
    # Home & lost
    hl = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[1],result=[2],type=ttype,create=False)
    # Home & tie
    ht = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[1],result=[3],type=ttype,create=False)
    # Home and no result
    hnr = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[1],result=[5],type=ttype,create=False)
    # Away and won
    aw = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[2],result=[1],type=ttype,create=False)
    #Away and lost
    al = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[2],result=[2],type=ttype,create=False)
    # Away and tie
    at = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[2],result=[3],type=ttype,create=False)
    # Away and no result
    anr = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[2],result=[5],type=ttype,create=False)    
    
    # Neutal and won
    nw = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[3],result=[1],type=ttype,create=False)
    # Neutral and lost
    nl = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[3],result=[2],type=ttype,create=False)
    # Neutral and tie
    nt = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[3],result=[3],type=ttype,create=False)
    # Neutral and no result
    nnr = getPlayerDataOD(profile=profileNo,dir=tdir,file=tfile,homeOrAway=[3],result=[5],type=ttype,create=False)    
    # Set the values as follows

    hw['ha'] = 1
    hw['result'] = 1
    
    hl['ha'] = 1
    hl['result'] = 2
    
    ht['ha'] = 1
    ht['result'] = 3
    
    hnr['ha'] = 1
    hnr['result'] = 5
    
    aw['ha'] = 2
    aw['result'] = 1
    
    al['ha'] = 2
    al['result'] = 2
    
    at['ha'] = 2
    at['result'] = 3
    
    anr['ha'] =  2
    anr['result'] =  5
    
    nw['ha'] = 3
    nw['result'] = 1
    
    nl['ha'] = 3
    nl['result'] = 2
    
    nt['ha'] = 3
    nt['result'] = 3
    
    nnr['ha'] =  3
    nnr['result'] =  5
      
    if not os.path.exists(tdir):
        os.mkdir(dir)
        #print("Directory " , dir ,  " Created ")
    else:    
        pass
        #print("Directory " , dir ,  " already exists, writing to this folder")
    
    # Create path
    path= os.path.join(tdir,tfile)
        
    df= pd.concat([hw,hl,ht,hnr,aw,al,at,anr])
    
    # Write to file 
    df.to_csv(path,index=False)
    
    return(df)
    
    
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 28 Nov 2018
# Function : getPlayerDataTT
# This function gets the One Day data of batsman/bowler and returns the data frame. This data frame can
# stored for use in other functions
##########################################################################################
def getPlayerDataTT(profile,opposition="",host="",dir="./data",file="player001.csv",type="batting",
                         homeOrAway=[1,2,3],result=[1,2,3,5],create=True) :
    '''
    Get the Twenty20 International player data from ESPN Cricinfo based on specific inputs and store in a file in a given directory~
    
    Description
    
    Get the Twenty20 player data given the profile of the batsman/bowler. The allowed inputs are home,away, neutralboth and won,lost,tied or no result of matches. The data is stored in a <player>.csv file in a directory specified. This function also returns a data frame of the player
    
    Usage
    
    getPlayerDataTT(profile, opposition="",host="",dir = "./data", file = "player001.csv", 
    type = "batting", homeOrAway = c(1, 2, 3), result = c(1, 2, 3,5))
    Arguments
    
    profile	
    This is the profile number of the player to get data. This can be obtained from http://www.espncricinfo.com/ci/content/player/index.html. Type the name of the player and click search. This will display the details of the player. Make a note of the profile ID. For e.g For Virat Kohli this turns out to be 253802 http://www.espncricinfo.com/india/content/player/35263.html. Hence the profile for Sehwag is 35263
    opposition	
    The numerical value of the opposition country e.g.Australia,India, England etc. The values are Afghanistan:40,Australia:2,Bangladesh:25,England:1,Hong Kong:19,India:6,Ireland:29, New Zealand:5,Pakistan:7,Scotland:30,South Africa:3,Sri Lanka:8,United Arab Emirates:27, West Indies:4, Zimbabwe:9; Note: If no value is entered for opposition then all teams are considered
    host	
    The numerical value of the host country e.g.Australia,India, England etc. The values are Australia:2,Bangladesh:25,England:1,India:6,New Zealand:5, South Africa:3,Sri Lanka:8,United States of America:11,West Indies:4, Zimbabwe:9 Note: If no value is entered for host then all host countries are considered
    dir	
    Name of the directory to store the player data into. If not specified the data is stored in a default directory "./data". Default="./data"
    file	
    Name of the file to store the data into for e.g. kohli.csv. This can be used for subsequent functions. Default="player001.csv"
    type	
    type of data required. This can be "batting" or "bowling"
    homeOrAway	
    This is vector with either or all 1,2, 3. 1 is for home 2 is for away, 3 is for neutral venue
    result	
    This is a vector that can take values 1,2,3,5. 1 - won match 2- lost match 3-tied 5- no result
    Details
    
    More details can be found in my short video tutorial in Youtube https://www.youtube.com/watch?v=q9uMPFVsXsI
    
    Value
    
    Returns the player's dataframe
    
    Note
    
    Maintainer: Tinniam V Ganesh <tvganesh.85@gmail.com>
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.wordpress.com/
    
    See Also
    
    bowlerWktRateTT getPlayerData
    
    Examples
    
    ## Not run: 
    # Only away. Get data only for won and lost innings
    kohli =getPlayerDataTT(253802,dir="../cricketr/data", file="kohli1.csv",
    type="batting")
    
    # Get bowling data and store in file for future
    ashwin = getPlayerDataTT(26421,dir="../cricketr/data",file="ashwin1.csv",
    type="bowling")
    
    kohli =getPlayerDataTT(253802,opposition = 2,host=2,dir="../cricketr/data", 
    file="kohli1.csv",type="batting")
    

    '''

    # Initial url to ""
    url =""
    suburl1 = "http://stats.espncricinfo.com/ci/engine/player/"
    suburl2 ="?class=3;"
    suburl3 = "template=results;"
    suburl4 = "view=innings"
    
    #Set opposition
    theOpposition = "opposition=" + opposition + ";"
    
    # Set host country
    hostCountry = "host=" + host + ";"
    
    # Create a profile.html with the profile number
    player = str(profile) + ".html"
       
    
    # Set the home or away
    str1=str2=""
    #print(len(homeOrAway))
    for i  in homeOrAway:
        if i == 1:
             str1 = str1 + "home_or_away=1;"
        elif i == 2:
             str1 = str1 + "home_or_away=2;"
        elif i == 3:
             str1 = str1 + "home_or_away=3;"
    HA= str1
    
    # Set the type batting or bowling
    t = "type=" + type + ";"
    
    # Set the result based on input
    str2=""
    for i in result:    
        if i == 1:
            str2 = str2+ "result=1;"        
        elif i == 2:
            str2 = str2 + "result=2;"          
        elif i == 3:
            str2 = str2 + "result=3;"
        elif i == 5:
            str2 = str2 + "result=5;"
    
    result =  str2 
    
    # Create composite URL
    url = suburl1 + player + suburl2 + hostCountry + theOpposition + HA + result + suburl3 + t + suburl4
    #print(url)
    # Read the data from ESPN Cricinfo
    dfList= pd.read_html(url)
    
    # Choose appropriate table from list of returned tables
    df=dfList[3]
    colnames= df.columns
    # Select coiumns based on batting or bowling
    if type=="batting" : 
        # Select columns [1:9,11,12,13]
        cols = list(range(0,9))
        cols.extend([10,11,12])
    elif type=="bowling":
        # Check if there are the older version of 8 balls per over (BPO) column
        # [1:8,10,11,12]
        
        # Select BPO column for older bowlers
        if colnames[1] =="BPO":
            # [1:8,10,11,12]
             cols = list(range(0,9))
             cols.extend([10,11,12])
        else:
            # Select columns [1:7,9,10,11]
             cols = list(range(0,8))
             cols.extend([8,9,10])
    
    
    #Subset the necessary columns
    df1 = df.iloc[:, cols]
    
    if not os.path.exists(dir):
        os.mkdir(dir)
        #print("Directory " , dir ,  " Created ")
    else:    
        pass
        #print("Directory " , dir ,  " already exists, writing to this folder")
    
    # Create path
    path= os.path.join(dir,file)
    
    if create:
        # Write to file 
        df1.to_csv(path)

    # Return the data frame
    return(df1)
    
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 18 Jun 2019
# Function: getTeamNumber
# This function returns the team number for a team name for Tests, ODIs and T20s
#
###########################################################################################

def getTeamNumber(teamName, matchType):
    '''
    Get the number of the Team
    
    Description
    
    This function returns the number of the Team for which analysis is to be done
    
    Usage
    
    getTeamNumber(teamName,matchType)
    Arguments
    
    teamName	
    The name of the team e.g Australia, India, Ghana etc
    matchType	
    The match type - Test, ODI or T20
    Value
    
    The numerical value of the team
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.in/
    See Also
    
    teamWinLossStatusVsOpposition teamWinLossStatusAtGrounds plotTimelineofWinsLosses
    
    Examples
    
    ## Not run: 
    #Get the team data for India for Tests
    
    teamNumber =getTeamNumber(teamName="India",matchType="Test")
    ## End(Not run)
    '''
    # Match type is Test
    if (matchType == "Test"):
        # Test teams
        teams = {"Afghanistan": 40, "Australia": 2, "Bangladesh": 25, "England": 1, "ICC World X1": 40,
                 "India": 6, "Ireland": 29, "New Zealand": 5, "Pakistan": 7, "South Africa": 3,
                 "Sri Lanka": 8, "West Indies": 4, "Zimbabwe": 9}
    
    # Match type is ODI
    elif (matchType == "ODI"):
        # ODI teams
        teams = {"Afghanistan": 40, "Africa XI": 4058, "Asia XI": 106, "Australia": 2,
                 "Bangladesh": 25, "Bermuda": 12, "England": 1, "ICC World X1": 140,
                 "India": 6, "Ireland": 29, "New Zealand": 5, "Pakistan": 7, "South Africa": 3,
                 "Sri Lanka": 8, "West Indies": 4, "Zimbabwe": 9, "Canada": 17, "East Africa": 14,
                 "Hong Kong": 19, "ICC World XI": 140, "Ireland": 29, "Kenya": 26, "Namibia": 28,
                 "Nepal": 32, "Netherlands": 15, "Oman": 37, "Papua New Guinea": 20, "Scotland": 30,
                 "United Arab Emirates": 27, "United States of America": 11, "Zimbabwe": 9}
    
    
    elif (matchType == "T20"):
        # T20 Teams
        teams = {"Afghanistan": 40, "Australia": 2, "Bahrain": 108, "Bangladesh": 25,
                 "Belgium": 42, "Belize": 115, "Bermuda": 12, "Botswana": 116, "Canada": 17, "Costa Rica": 4082,
                 "Germany": 35, "Ghana": 135, "Guernsey": 1094, "Hong Kong": 19, "ICC World X1": 140,
                 "India": 6, "Ireland": 29, "Italy": 31, "Jersey": 4083, "Kenya": 26, "Kuwait": 38,
                 "Maldives": 164, "Malta": 45, "Mexico": 165, "Namibia": 28, "Nepal": 32, "Netherlands": 15,
                 "New Zealand": 5, "Nigeria": 173, "Oman": 37, "Pakistan": 7, "Panama": 183, "Papua New Guinea": 20,
                 "Philippines": 179, "Qatar": 187, "Saudi Arabia": 154, "Scotland": 30, "South Africa": 3,
                 "Spain": 200, "Sri Lanka": 8, "Uganda": 34, "United Arab Emirates": 27,
                 "United States of America": 11, "Vanuatu": 216, "West Indies": 4}
    else:
        print("Unknown match ")
    
    # Return team number
    return (teams[teamName])

##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 18 Jun 2019
# Function: getMatchType
# This function returns the match number for a  match type viz. Tests, ODIs and T20s
#
###########################################################################################

def getMatchType(matchType):
      '''
      getMatchType(matchType)
      Arguments
    
      matchType	
      The match type - Test, ODI or T20
      Value
    
      The numerical value of match type
    
      Note
    
      Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
      Author(s)
    
      Tinniam V Ganesh
    
      References
    
      http://www.espncricinfo.com/ci/content/stats/index.html
      https://gigadom.in/
      See Also
    
      teamWinLossStatusVsOpposition teamWinLossStatusAtGrounds plotTimelineofWinsLosses
    
      Examples
    
      ## Not run: 
      #Get the team data for India for Tests
    
      match =getMatchType("Test")
    
      ## End(Not run)
      '''     
      match = {"Test": 1, "ODI": 2, "T20": 3}
      return (match[matchType])
    
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 18 Jun 2019
# Function: getTeamData
# This function returns the team data as a CSV file and data frame for a given matchType
# homeOrAway and result vectors. This can be done for a given team as 'bat' or 'bowl'
# option
#
###########################################################################################
def getTeamData(teamName,dir=".",file="team001.csv",matchType="Test",
                        homeOrAway=[1,2,3],result=[1,2,3,4],teamView="bat",save=False) :
    '''
    Get the data for a team in a match type viz.for Test, ODI and T20
    
    Description
    
    This function returns team data as a CSV file and/or a dataframe for Test, ODI and T20
    
    Usage
    
    getTeamData(dir=".",file="team001.csv",matchType="Test",
            homeOrAway=[1,2,3],result=[1,2,3,4],teamView="bat",save=False,teamName)
    Arguments
    
    dir	
    The directory where the team data CSV file be saved
    file	
    The name of the CSV file to save to
    matchType	
    The match type - Test, ODI , T20
    homeOrAway	
    Whether the data has to be got for home-1, away(overseas)-2 or neutral -3
    result	
    The result of the match for which data is to be saved - won-1, lost -2, tied-3, draw-4
    teamView	
    This can be 'bat' - batting team or 'bowl' - bowling team
    save	
    This can be set as True or False
    teamName	
    This is team name
    Value
    
    The required data frame
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.in/
    See Also
    
    teamWinLossStatusVsOpposition teamWinLossStatusAtGrounds plotTimelineofWinsLosses
    
    Examples
    
    ## Not run: 
    #Get the team data for India for Tests
    
    df=getTeamData(dir=".",file="australia.csv", matchType="Test",homeOrAway=[1,2,3],
           result=[1,2,3, 4],teamView='bat',teamName="Australia")
    
    ## End(Not run)
    
    '''
    
    # Initialize url to ""
    url = ""
    suburl1 = "http://stats.espncricinfo.com/ci/engine/stats/index.html"
    
    # Get the match tyoe
    match = getMatchType(matchType)
    # Create suburl2
    suburl2 = "?class=" + str(match) + ";"
    
    suburl3 = "template=results;"
    suburl4 = "type=team;"
    suburl5 = "size=200;"
    suburl6 = "view=innings;"
    
    # Set the home or away depending on the values in homeOrAway vector
    str1 = str2 = str3 = ""
    if (1 in homeOrAway):
        str1 = "home_or_away=1;"
    
    if (2 in homeOrAway):
        str2 = "home_or_away=2;"
    
    if (3 in homeOrAway):
        str3 = "home_or_away=3;"
    
    HA = str1 + str2 + str3
    
    # Set the result based on  result vector
    str1 = str2 = str3 = str4 = ""
    if (1 in result):
        str1 = "result=1;"
    
    if (2 in result):
        str2 = "result=2;"
    
    if (3 in result):
        str3 = "result=3;"
    
    # Test has result 'draw-4' while ODI and T20 have result 'no result-5'
    if (matchType == "Test"):  # Test
        if (4 in result):
            str4 = "result=4;"
    else:  # ODI & T20
        if (5 in result):
            str4 = "result=5;"
    
    result = str1 + str2 + str3 + str4
    # Set the team
    
    # Get the team number
    team = getTeamNumber(teamName, matchType)
    
    # Set the team for which data is required
    theTeam = "team=" + str(team) + ";"
    
    # Set the data view
    dataView = "team_view=" + teamView + ";"
    
    # Create composite URL
    baseurl = suburl1 + suburl2 + suburl3 + suburl4 + suburl5 + suburl6 + HA + result + theTeam + dataView
    
    notDone = True
    i = 0
    teamDF = pd.DataFrame()
    
    # Loop through the data 200 (max)rows at a time
    while (notDone):
        i = i + 1
        page = ""
        page = "page=" + str(i)
        url = baseurl + page

        dfList = pd.read_html(url)
        df = dfList[2]
    
        if (df.shape[0] == 1):
            break;
        else:
            teamDF = pd.concat([teamDF, df])
    
        if not os.path.exists(dir):
            os.mkdir(dir)
            # print("Directory " , dir ,  " Created ")
        else:
            pass
            # print("Directory " , dir ,  " already exists, writing to this folder")
    
        # Create path
        path = os.path.join(dir, file)
    
        if save:
            # Write to file 
            teamDF.to_csv(path)
    
    return (teamDF)


##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 18 Jun 2019
# Function: getTeamData
# This function returns the team data as a CSV file and data frame for a given matchType
# homeOrAway and result vectors along with a column which includes whether the match was
#home, away, neutral zone etc. This can be done for a given team as 'bat' or 'bowl'
# option
#
###########################################################################################

def getTeamDataHomeAway(teamName,dir=".",teamView="bat",matchType="Test",file="team001HA.csv",save=True):
    '''
    Get the data for a team in a match type viz.for Test, ODI and T20 with the home/overseas/neutral
    
    Description
    
    This function returns team data as a CSV file and/or a dataframe for Test, ODI and T20 with an additional column showing home, away or neutral venue where the match was played
    
    Usage
    
    getTeamDataHomeAway(dir=".",teamView="bat",matchType="Test",file="team001HA.csv",
         save=TRUE,teamName)
    Arguments
    
    dir	
    The directory where the team data CSV file be saved
    teamView	
    Team view can be either 'bat' (batting team) or 'bowl' (bowling team)
    matchType	
    The match type - Test, ODI , T20
    file	
    The name of te file to save to
    save	
    This can be TRUE or FALSE
    teamName	
    Team name is the team namely - Australia, India, England etc
    Value
    
    The required data frame
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.in/
    See Also
    
    teamWinLossStatusVsOpposition teamWinLossStatusAtGrounds plotTimelineofWinsLosses
    
    Examples
    
    ## Not run: 
    #Get the team data for India for Tests
    
    getTeamDataHomeAway(teamName="India",file="india.csv")
    
    ## End(Not run)
    
    '''
    
    print("Working...")
    # Check if match type is Test. The result is won, lost, draw or tie
    if (matchType == "Test"):
        df1 = getTeamData(teamName=teamName, dir=".", file="team001.csv", matchType=matchType, homeOrAway=[1],
                          result=[1, 2, 3, 4], teamView=teamView)
        df2 = getTeamData(teamName=teamName, dir=".", file="team001.csv", matchType=matchType, homeOrAway=[2],
                          result=[1, 2, 3, 4], teamView=teamView)
        df3 = getTeamData(teamName=teamName, dir=".", file="team001.csv", matchType=matchType, homeOrAway=[3],
                          result=[1, 2, 3, 4], teamView=teamView)
    else:  # ODI & T20. The result is won, lost, tie or no result
    
        df1 = getTeamData(teamName=teamName, dir=".", file="team001.csv", matchType=matchType, homeOrAway=[1],
                          result=[1, 2, 3, 5], teamView=teamView)
        df2 = getTeamData(teamName=teamName, dir=".", file="team001.csv", matchType=matchType, homeOrAway=[2],
                          result=[1, 2, 3, 5], teamView=teamView)
        df3 = getTeamData(teamName=teamName, dir=".", file="team001.csv", matchType=matchType, homeOrAway=[3],
                          result=[1, 2, 3, 5], teamView=teamView)
    

    # Create the column values home, away or neutral
    df1['ha'] = "home"
    df2['ha'] = "away"
    df3['ha'] = "neutral"
    
    df = pd.DataFrame()
    
    # Stack the rows to create dataframe
    if (df1.shape[0] != 0):
        df = pd.concat([df, df1])
    if (df2.shape[0] != 0):
        df = pd.concat([df, df2])
    if (df3.shape[0] != 0):
        df = pd.concat([df, df3])
    
    # Create path
    path = os.path.join(dir, file)
    
    if save:
        # Write to file 
        df.to_csv(path)
    
    return (df)
   
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 18 Jun 2019
# Function: cleanTeamData
# This function cleans the team data for Test, ODI and T20
#
###########################################################################################

def cleanTeamData(df,matchType):
    '''
    Clean the team data for Test, ODI and T20
    
    Description
    
    This function cleans the team data for Test, ODI and T20
    
    Usage
    
    cleanTeamData(df,matchType)
    Arguments
    
    df	
    Data frame
    matchType	
    Match type - Test, ODI, T20
    Value
    
    The cleaned Data frame
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.in/
    See Also
    
    teamWinLossStatusVsOpposition teamWinLossStatusAtGrounds plotTimelineofWinsLosses
    
    Examples
    
    ## Not run: 
    #Get the team data for India for Tests
    df =getTeamDataHomeAway(file="india.csv",teamName="India",matchType='Test')
    df1 =cleanTeamData(df,"Test")
    
    ## End(Not run)
    
    '''
    # Remove rows with 'DNB
    
    a = df.Score != 'DNB'
    df1 = df[a]
    
    if (matchType == "Test"):
        # Remove columns 0,8 & 12 for Tests. They have empty columns
        cols = [0, 8, 12]
        df2 = df1.drop(df1.columns[cols], axis=1)
    # Remove columns 8 & 12 for ODI and T20. They have empty columns
    elif ((matchType == "ODI") or (matchType == "T20")):
        cols = [0, 7, 11]
        df2 = df1.drop(df1.columns[cols], axis=1)
    
    # Fix the Opposition column, remove "^ v"
    df2.Opposition = df2.Opposition.str.replace("v ", "")
    # Return cleaned dataframe
    return (df2)

      
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 18 Jun 2019
# Function: teamWinLossStatusVsOpposition
# This function returns a team's win/loss/draw/tie status against the opposition. The
# matches could be played at home/away/neutral venues for Test, ODI and T20
#
###########################################################################################

def teamWinLossStatusVsOpposition(file,teamName,opposition=["all"],homeOrAway=["all"],matchType="Test",plot=False):
    '''
    Compute the wins/losses/draw/tied etc for a Team in Test, ODI or T20 against opposition
    
    Description
    
    This function computes the won,lost,draw,tied or no result for a team against other teams in home/away or neutral venues and either returns a dataframe or plots it against opposition
    
    Usage
    
    teamWinLossStatusVsOpposition(file,teamName,opposition=["all"],homeOrAway=["all"],
          matchType="Test",plot=FALSE)
    Arguments
    
    file	
    The CSV file for which the plot is required
    teamName	
    The name of the team for which plot is required
    opposition	
    Opposition is a list namely ["all"] or ["Australia", "India", "England"]
    homeOrAway	
    This parameter is a list which is either ["all"] or a vector of venues ["home","away","neutral")]
    matchType	
    Match type - Test, ODI or T20
    plot	
    If plot=False then a data frame is returned, If plot=TRUE then a plot is generated
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.in/
    See Also
    
    teamWinLossStatusVsOpposition teamWinLossStatusAtGrounds plotTimelineofWinsLosses
    
    Examples
    
    ## Not run: 
    #Get the team data for India for Tests
    
    df <- getTeamDataHomeAway(teamName="India",file="indiaOD.csv",matchType="ODI")
    teamWinLossStatusAtGrounds("india.csv",teamName="India",opposition=c("Australia","England","India"),
                              homeOrAway=c("home","away"),plot=TRUE)
    
    ## End(Not run)
    
    '''
    # Read CSV file
    df = pd.read_csv(file)
    # Clean data
    df1 = cleanTeamData(df, matchType)

    
    # Get the list of countries in opposition and filter those rows
    if ("all" in opposition):
        # Do not filter
        pass
    else:
        df1 = df1[df1['Opposition'].isin(opposition)]
    
    # Check home/away/neutral from list homeOrAway and filter rows
    if ("all" in homeOrAway):
        # Do not filter
        pass
    else:
        df1 = df1[df1['ha'].isin(homeOrAway)]
    
    # Select columns, group and count
    df2=df1.groupby(['Opposition','Result','ha']).Opposition.agg('count').to_frame('count').unstack().fillna(0)['count']
    
    # If plot is True
    if (plot == True):
        # Collapse list of countries in opposition
        separator = '-'
        oppn = separator.join(opposition)
        # Collapse vectors of homeOrAway vector
        ground = separator.join(homeOrAway)
    
        atitle = "Win/Loss status of " + teamName + " against opposition in " + matchType + "(s)"
    
        asub = "Against " + oppn + " teams at " + ground + " grounds"
    

        # Plot for opposition and home/away for a team in Tes, ODI and T20
        df2.plot(kind='bar',stacked=False,legend=True,fontsize=8,width=1)
    
        plt.xlabel('Opposition')
        plt.ylabel('Win/Loss count')
        plt.suptitle(atitle)
        plt.title(asub)
        plt.text(5, 10,'Data source-Courtesy:http://cricsheet.org',
         horizontalalignment='center',
         verticalalignment='center',
         )
        plt.subplots_adjust( bottom=0.9)
        plt.show()
        plt.gcf().clear()
    else:
        # Return dataframe
        return (df2)

##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 18 Jun 2019
# Function: teamWinLossStatusAtGrounds
# This function returns a team's win/loss/draw/tie status against the opposition at venues.
# The matches could be played at home/away/neutral venues for Test, ODI and T20s. The output is either a
# dataframe or a plot
#
###########################################################################################

def teamWinLossStatusAtGrounds(file,teamName,opposition=["all"],homeOrAway=["all"],matchType="Test",plot=False):
    '''
    Compute the wins/losses/draw/tied etc for a Team in Test, ODI or T20 at venues
    
    Description
    
    This function computes the won,lost,draw,tied or no result for a team against other teams in home/away or neutral venues and either returns a dataframe or plots it for grounds
    
    Usage
    
    teamWinLossStatusAtGrounds(file,teamName,opposition=["all"],homeOrAway=["all"],
                  matchType="Test",plot=FALSE)
    Arguments
    
    file	
    The CSV file for which the plot is required
    teamName	
    The name of the team for which plot is required
    opposition	
    Opposition is a vector namely ["all")] or ["Australia", "India", "England"]
    homeOrAway	
    This parameter is a vector which is either ["all")] or a vector of venues ["home","away","neutral"]
    matchType	
    Match type - Test, ODI or T20
    plot	
    If plot=FALSE then a data frame is returned, If plot=TRUE then a plot is generated
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.in/
    See Also
    
    teamWinLossStatusVsOpposition teamWinLossStatusAtGrounds plotTimelineofWinsLosses
    
    Examples
    
    ## Not run: 
    #Get the team data for India for Tests
    
    df <- getTeamDataHomeAway(teamName="India",file="indiaOD.csv",matchType="ODI")
    teamWinLossStatusAtGrounds("india.csv",teamName="India",opposition=c("Australia","England","India"),
                              homeOrAway=c("home","away"),plot=TRUE)
    
    ## End(Not run)
    
    '''
    # Read CSV file
    df = pd.read_csv(file)
    # Clean data
    df1 = cleanTeamData(df, matchType)
    
    # Get the list of countries in opposition and filter those rows
    if ("all" in opposition):
        # Do not filter
        pass
    else:
        df1 = df1[df1['Opposition'].isin(opposition)]
    
    # Check home/away/neutral from list homeOrAway and filter rows
    if ("all" in homeOrAway):
        # Do not filter
        pass
    else:
        df1 = df1[df1['ha'].isin(homeOrAway)]
    
    # Select columns, group and count
    df2=df1.groupby(['Ground','Result','ha']).Opposition.agg('count').to_frame('count').unstack().fillna(0)['count']
    
    # If plot is True
    if (plot == True):
        # Collapse list of countries in opposition
        separator = '-'
        oppn = separator.join(opposition)
        # Collapse vectors of homeOrAway vector
        ground = separator.join(homeOrAway)
    
        atitle = "Win/Loss status of " + teamName + " against opposition in " + matchType + "(s)"
    
        asub = "Against " + oppn + " teams at " + ground + " grounds"
    
        # Plot for opposition and home/away for a team in Tes, ODI and T20
        df2.plot(kind='bar',stacked=False,legend=True,fontsize=8,width=1)
        plt.xlabel('Ground')
        plt.ylabel('Win/Loss count')
        plt.suptitle(atitle)
        plt.title(asub)
        plt.text(5, 3,'Data source-Courtesy:http://cricsheet.org',
         horizontalalignment='center',
         verticalalignment='center',
         )
        plt.subplots_adjust( bottom=0.9)
        plt.show()
        plt.gcf().clear()
    else:
        # Return dataframe
        return (df2)
    
##########################################################################################
# Designed and developed by Tinniam V Ganesh
# Date : 18 Jun 2019
# Function: plotTimelineofWinsLosses
# This function plot the timelines of win/lost/draw/tie against opposition at
# venues for Test, ODI and T20s
#
###########################################################################################

def plotTimelineofWinsLosses(file,teamName,opposition=["all"],homeOrAway=["all"],
                                     startDate="2001-01-01",endDate="2019-01-01",matchType="Test"):
    '''
    Plot the time line of wins/losses/draw/tied etc for a Team in Test, ODI or T20
    
    Description
    
    This function returns plots a time line of won,lost,draw,tied or no result for a team against other teams in home/away or neutral venues
    
    Usage
    
    plotTimelineofWinsLosses(file,teamName,opposition=["all"],homeOrAway=["all"],
          startDate="2001-01-01",endDate="2019-01-01",matchType="Test")
    Arguments
    
    file	
    The CSV file for which the plot is required
    teamName	
    The name of the team for which plot is required
    opposition	
    Opposition is a vector namely ["all"] or ["Australia", "India", "England"]
    homeOrAway	
    This parameter is a vector which is either ["all"] or a vector of venues ["home","away","neutral"]
    startDate	
    The start date from which time line is required
    endDate	
    The end data for which the time line plot is required
    matchType	
    Match type - Test, ODI or T20
    Value
    
    None
    
    Note
    
    Maintainer: Tinniam V Ganesh tvganesh.85@gmail.com
    
    Author(s)
    
    Tinniam V Ganesh
    
    References
    
    http://www.espncricinfo.com/ci/content/stats/index.html
    https://gigadom.in/
    See Also
    
    teamWinLossStatusVsOpposition teamWinLossStatusAtGrounds plotTimelineofWinsLosses
    
    Examples
    
    ## Not run: 
    #Get the team data for India for Tests
    
    df <- getTeamDataHomeAway(teamName="India",file="indiaOD.csv",matchType="ODI")
    plotTimelineofWinsLosses("indiaOD.csv",teamName="India",
            startDate="2015-01-01",endDate="2019-01-01", matchType="ODI")
    
    ## End(Not run)
    
    '''
    # Read CSV file
    df = pd.read_csv(file)
    # Clean data
    df1 = cleanTeamData(df, matchType)
    
    # Get the list of countries in opposition and filter those rows
    if ("all" in opposition):
        # Do not filter
        pass
    else:
        df1 = df1[df1['Opposition'].isin(opposition)]
    
    # Check home/away/neutral from list homeOrAway and filter rows
    if ("all" in homeOrAway):
        # Do not filter
        pass
    else:
        df1 = df1[df1['ha'].isin(homeOrAway)]
    
    # FIlter won and set to 1
    a = df1.Result == "won"
    df2 = df1[a]
    if (df2.shape[0] != 0):
        df2['result'] = 1
    
    # Filter tie and set to 0.5
    a = df1.Result == "tie"
    df3 = df1[a]
    # No tie
    if (df3.shape[0] != 0):
        df3['result'] = 0.5
    
    # Test has w
    if (matchType == "Test"):
        # FIlter draw and set to -0.5
        a = df1.Result == "draw"
        df4 = df1[a]
        if (df4.shape[0] != 0):  # No draw
            df4['result'] = 0
    elif ((matchType == "ODI") or (matchType == "T20")):
        # FIlter 'no result' and set to 0
        a = df1.Result == "n/r"
        df4 = df1[a]
        if (df4.shape[0] != 0):
            df4['result'] = -0.5
    
    # Filter lost and set to -1
    a = df1.Result == "lost"
    df5 = df1[a]
    if (df5.shape[0] != 0):
        df5['result'] = -1
    
    df6 = pd.concat([df2, df3, df4, df5])

    df6['date'] = pd.to_datetime(df6['Start Date'])
    separator = '-'
    oppn = separator.join(opposition)
    # Collapse vectors of homeOrAway vector
    ground = separator.join(homeOrAway)
    
    atitle = "Timeline of Win/Loss status of " + teamName + " in " + matchType + "(s)"
    asub = "Against " + oppn + " teams at " + ground + " grounds"
    
    # Sort by Start date
    df7 = df6[['date', 'result']].sort_values(by='date')
    # Filter between start and end dates
    m = ((df7.date >= startDate) & (df7.date <= endDate))
    df8 = df7[m]
    if (matchType == "Test"):
        plt.plot(df8['date'], df8['result'])
        plt.axhline(y=1, color='r',linestyle="--")
        plt.axhline(y=0.5, color='b',linestyle="--")
        plt.axhline(y=0, color='y',linestyle="--")
        plt.axhline(y=-1, color='y',linestyle="--")
        plt.text(startDate, 1, 'Won')
        plt.text(startDate, .5, 'tie')
        plt.text(startDate, 0, 'draw')
        plt.text(startDate, -1, 'lost')
        plt.xlabel('Date')
        plt.ylabel('Win/Loss status')
        plt.suptitle(atitle)
        plt.title(asub)
        plt.subplots_adjust( bottom=0.9)
        plt.show()
        plt.gcf().clear()
    
    elif ((matchType == "ODI") or (matchType == "T20")):
        plt.plot(df8['date'], df8['result'])
        plt.axhline(y=1, color='r',linestyle="--")
        plt.axhline(y=0.5, color='b',linestyle="--")
        plt.axhline(y=-0.5, color='y',linestyle="--")
        plt.axhline(y=-1, color='y',linestyle="--")
        plt.text(startDate, 1, 'Won')
        plt.text(startDate, .5, 'tie')
        plt.text(startDate, -0.5, 'no result')
        plt.text(startDate, -1, 'lost')
        plt.xlabel('Date')
        plt.ylabel('Win/Loss status')
        plt.suptitle(atitle)
        plt.title(asub)
        plt.subplots_adjust( bottom=0.9)
        plt.show()
        plt.gcf().clear()