import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dataSet = [
    [10 , "YES" , 160 , 2 , "NO" ,"DON'T BUY"],
    [9 , "YES" , 160 , 2 , "NO" ,"DON'T BUY"],
    [8 , "YES" , 160 , 2 , "NO" ,"DON'T BUY"],
    [7 , "NO" , 160 , 2 , "NO" ,"DON'T BUY"],
    [10 , "YES" ,170 , 2 , "NO" ,"DON'T BUY"],
    [9 , "NO" , 180 , 2.2 , "NO" ,"DON'T BUY"],
    [8 , "YES" , 190 , 1.2 , "NO" ,"DON'T BUY"],
    [7 , "YES" , 200 , 1.3 , "NO" ,"DON'T BUY"],
    [10 , "NO" ,210 , 2.1 , "NO" ,"DON'T BUY"],
    [9 , "YES" , 170 , 1.5 , "YES" ,"DON'T BUY"],
    [8 , "YES" , 170 , 1.6 , "NO" ,"DON'T BUY"],
    [7 , "YES" , 170 , 1.4 , "NO" ,"DON'T BUY"],
    [10 , "YES" , 160 , 2.3 , "YES" ,"DON'T BUY"],
    [10 , "YES" , 160 , 2.4, "NO" ,"DON'T BUY"],
    [10 , "YES" , 160 , 2.5 , "YES" ,"DON'T BUY"],
    [10 , "YES" , 160 , 2.7 , "YES" ,"DON'T BUY"],
    [8 , "Yes" , 200 , 5 , "NO" ,"DON'T BUY"],
    [3 , "NO" , 20 , 0.1 , "YES" , "BUY"],
    [4 , "NO" , 30 , 0.2 , "YES" , "BUY"],
    [5 , "NO" , 40 , 0.3 , "YES" , "BUY"],
    [3 , "NO" , 50 , 0.4 , "YES" , "BUY"],
    [4 , "YES" , 60 , 0.5 , "YES" , "BUY"],
    [5 , "NO" , 70 , 0.6 , "NO" , "BUY"],
    [6 , "NO" , 80 , 0.7 , "YES" , "BUY"],
    [3 , "YES" , 90 , 0.8 , "YES" , "BUY"],
    [4 , "NO" , 100 , 0.9 , "NO" , "BUY"],
    [5 , "NO" , 110 , 1 , "YES" , "BUY"],
    [3 , "NO" , 120 , 0.1 , "YES" , "BUY"],
    [4 , "YES" , 130 , 0.2 , "NO" , "BUY"],
    [5 , "NO" , 140 , 0.3 , "YES" , "BUY"],
    [6 , "NO" , 80 , 0.4 , "NO" , "BUY"],
    [6 , "NO" , 20 , 0.5 , "YES" , "BUY"]
]

header = ["taille liste" , "presence educorant" , "valeur calorique" , "taux sel",
          "tous les graisses de meilleur qualité" , "label"]





#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(features,labels)
#p = clf.predict([[10, 1, 6, 20, 2, 0]])
#print(p)

def class_counts(rows):

    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts
#print(class_counts(dataSet))

def unique_vals(rows, col):
    return set([row[col] for row in rows])
#print(unique_vals(dataSet,0)) retourne {10,3} cad la premiere colonne

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))
#q = Question(1,"YES")
#example = dataSet[0]
#print(q.match(example))

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows
#true , false =partition(dataSet,Question(1,"YES"))
#print(true)
#print(false)


def gini(rows):
  #calcul d'incertitude
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    # print("p =" +str(p))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


#current = gini(dataSet)
#true , false =partition(dataSet,Question(1,"YES"))
#print(info_gain(true,false,current))

def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1
    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question
best_gain, best_question = find_best_split(dataSet)


#print(best_gain)
#print(best_question)

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class DecisionNode:
    def __init__(self,question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return DecisionNode(question, true_branch, false_branch)

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

my_tree = build_tree(dataSet)
#print(classify( [6 , "YES" , 1 , 200 , 9 , "NO" ,"DON'T BUY"], my_tree))

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

#header = ["taille liste" , "presence educorant" , "valeur calorique" , "taux sel",
  #  "tous les graisses de meilleur qualité" , "label"]


print(print_leaf(classify( [3 , "YES" , 90 , 0.8 , "YES"] , my_tree)))
print(print_leaf(classify([ 8 , "YES" , 160 , 2 , "NO"  ], my_tree)))
print(print_leaf(classify( [2 , "NO" , 20 , 0.1 , "YES" ], my_tree)))
print(print_leaf(classify( [3 , "NO" , 170 , 1.5 , "NO" ], my_tree)))