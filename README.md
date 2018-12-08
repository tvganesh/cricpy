# Cricpy

[![PyPI](https://img.shields.io/pypi/v/cricpy.svg)](https://pypi.org/project/cricpy/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cricpy.svg)](https://pypi.org/project/cricpy/) [![PyPI - Wheel](https://img.shields.io/pypi/wheel/cricpy.svg)](https://pypi.org/project/cricpy/)

#### Package downloads from the Pypi</b>

[![Downloads](https://pepy.tech/badge/cricpy)](https://pepy.tech/project/cricpy)

[![Downloads](https://pepy.tech/badge/cricpy/month)](https://pepy.tech/project/cricpy)

[![Downloads](https://pepy.tech/badge/cricpy/week)](https://pepy.tech/project/cricpy)
<hr>

## Description 
Tools for analyzing performances of cricketers based on stats in ESPN Cricinfo Statsguru. The toolset can  be used for analysis of Tests,ODI amd Twenty20 matches of both batsmen and bowlers.

## Installation 

```py
# Install the package
pip install cricpy
```
## Importing and Loading Data 

`cricpy` can be imported using the typical `import`:

```py
# Import cricpy
import cricpy
```

To load the data, You could either do specific import


```py
#1.  
import cricpy.analytics as ca 
#ca.batsman4s("../dravid.csv","Rahul Dravid")

```
or import all the functions

```py
#2.

from cricpy.analytics import *
#batsman4s("../dravid.csv","Rahul Dravid")
```

## References

* [Introducing cricpy:A python package to analyze performances of cricketers](https://gigadom.in/2018/10/28/introducing-cricpya-python-package-to-analyze-performances-of-cricketrs/)


## R-Package

If you are an R-user, There is an R equivalent of `cricpy` - that is [`cricketr`](http://tvganesh.github.io/cricketer/cricketer.html).

