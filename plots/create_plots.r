#Full Train
cn2_acc = 0.8
expCN2_good_many_acc = 0.85
expCN2_good_few_acc =  0.825
expCN2_bad_many_acc = 0.35
expCN2_bad_few_acc = 0.45

# Grouped Bar Plot
scores <- c(cn2_acc, expCN2_good_many_acc, expCN2_good_few_acc, expCN2_bad_many_acc, expCN2_bad_few_acc)

full_train = barplot(scores,legend = c("CN2", "4 Good", "1 Good", "4 Bad", "1 Bad"),
        col = c("blue", "red", "green", "orange", "pink"),
        names.arg=c("CN2", "4 Good", "1 Good", "4 Bad", "1 Bad"),
        xlab = "Performance with complete train set",
        ylim = range(0,1))

text(full_train, 0.1, round(scores, 2), cex=1.0)


#PArtial Train
cn2_acc = 0.725
expCN2_good_many_acc = 0.825
expCN2_good_few_acc =  0.75
expCN2_bad_many_acc = 0.35
expCN2_bad_few_acc = 0.425
