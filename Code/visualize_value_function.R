# This script plots the value function of a single leg flight example in a threedimensional plot V(t, c)

# input pickle object (from Python)
library(reticulate)

# data structure (dataframe)
library(data.table)
library(tidyverse)

# plotting
# library(ggplot2)
# library(rgl)
# library(scatterplot3d)
library(plotly)
library(RSelenium)

# read data
setwd("C:/Users/Stefan/LRZ Sync+Share/Masterarbeit-Klein/Code")
source_python("pickle_reader.py")
dat <- read_pickle_file("C:/Users/Stefan/LRZ Sync+Share/Masterarbeit-Klein/Code/Results/smallTest2-False-DP-190619-1333/totalresults.data")

# data is read as one dimensional list

# get first final result set
d = dat[[1]]

# transform it to one large dataframe
l = list()
for (c in 1:length(d)){
  l[[c]] <- data.frame("c" = c-1, "t" = seq.int(nrow(d[[c]]))-1, "v" = d[[c]]$value)
}
a = rbindlist(l)

# transform dataframe to format used for plotting (x and y vector, z matrix)
m <- a %>% spread(key = c, value = v)
m$t <- NULL
c <- unique(a$c)
t <- unique(a$t)
v <- t %o% c
v <- data.matrix(m)

# plot (export not working on 19.6.2019)
plot_ly(x=c, y=t, z=v, type = "surface") %>%
  export(file = "value_function.svg", selenium = RSelenium::rsDriver(browser = "chrome"))
