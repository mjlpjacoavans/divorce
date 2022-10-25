# Webapp imports
import random
import string
import threading
import time
import re

import numpy
from flask import Flask, render_template, request, redirect

# Tree imports
import pandas as pd
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from random import randint

# NN imports
import tensorflow
import keras

app = Flask(__name__)
classifier, accuracy = None, None
admin_password = "admin"
tree_seed = 46841

# all_questions = [
# 	"My spouse and I have similar ideas about how roles should be in marriage",
# 	"I enjoy traveling with my wife.",
# 	"The time I spent with my wife is special for us.",
# 	"I know my spouse's basic anxieties.",
# 	"My discussion with my spouse is not calm.",
# 	"I know my spouse's favorite food.",
# 	"I know my spouse's friends and their social relationships.",
# 	"I know what my spouse's current sources of stress are.",
#
#
# 	# "Even if I'm right in the discussion, I stay silent to hurt my spouse."
# ]



all_questions = [
	"We're just starting a discussion before I know what's going on.",
	'We share the same views about being happy in our life with my spouse',
	'My spouse and I have similar ideas about how roles should be in marriage',
	'My spouse and I have similar ideas about how marriage should be',
	'I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.',
	'I enjoy traveling with my wife.',
	'My spouse and I have similar values in trust.',
	'Our dreams with my spouse are similar and harmonious.',
	'Our discussions often occur suddenly.',
	'When I talk to my spouse about something, my calm suddenly breaks.',
	'The time I spent with my wife is special for us.',
	'I know my spouse very well.',
	'I can be humiliating when we discussions.',
	"We're compatible with my spouse about what love should be.",
	"I hate my spouse's way of open a subject.",
	"I know my spouse's friends and their social relationships.",
	"I know my spouse's basic anxieties.",
	"I know what my spouse's current sources of stress are.",
	'I enjoy our holidays with my wife.',
	'My spouse and I have similar values in terms of personal freedom.',
	'I know exactly what my wife likes.',
	'Most of our goals for people (children, friends, etc.) are the same.',
	'My discussion with my spouse is not calm.',
	'I can insult my spouse during our discussions.',
	"I can use negative statements about my spouse's personality during our discussions.",
	'If one of us apologizes when our discussion deteriorates, the discussion ends.',
	"I have knowledge of my spouse's inner world.",
	"Sometimes I think it's good for me to leave home for a while.",
	"I know my spouse's hopes and wishes.",
	'My spouse and I have similar sense of entertainment.',
	'I can tell you what kind of stress my spouse is facing in her/his life.',
	"I know my spouse's favorite food.",
	'I can use offensive expressions during our discussions.',
	'Most of our goals are common to my spouse.',
	'When discussing with my spouse, I usually use expressions such as ‘you always’ or ‘you never’ .',
	'I know how my spouse wants to be taken care of when she/he sick.',
	'I know we can ignore our differences, even if things get hard sometimes.',
	'When I discuss with my spouse, to contact him will eventually work.',
	"I'm not afraid to tell my spouse about her/his incompetence.",
	'When we need it, we can take our discussions with my spouse from the beginning and correct it.',
	'I feel aggressive when I argue with my spouse.',
	"I'm not actually the one who's guilty about what I'm accused of.",
	"I have nothing to do with what I've been accused of.",
	"When I argue with my spouse, ı only go out and I don't say a word.",
	'When I discuss, I remind my spouse of her/his inadequacy.',
	"I'm not the one who's wrong about problems at home.",
	'When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.',
	"I wouldn't hesitate to tell my spouse about her/his inadequacy.",
	'I feel right in our discussions.',
	'I mostly stay silent to calm the environment a little bit.',
	"I'd rather stay silent than discuss with my spouse.",
	'We are like two strangers who share the same environment at home rather than family.',
	"Even if I'm right in the discussion, I stay silent to hurt my spouse.",
	"We don't have time at home as partners."
	
][:10]


def prepare_dataset(dataset_file="./dataset/divorce.xlsx",
                    description_file="./dataset/refercence.tsv"):
	print("[*] Preparing dataset")
	headers_frame = pd.read_csv(description_file, sep="|")
	headers = list(headers_frame["description"])
	headers.append("Class")
	df = pd.read_excel(dataset_file, names=headers)

	X, y = df[all_questions], df["Class"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	return df, X_train, X_test, y_train, y_test


df, X_train, X_test, y_train, y_test = prepare_dataset(
	".\\dataset\\divorce.xlsx", ".\\dataset\\reference.tsv")


def train_tree():
	print(f"[*] Creating descision tree with seed: {tree_seed}")
	dtree = DecisionTreeClassifier(random_state=tree_seed, max_depth=5)
	_tree_hist = dtree.fit(X_train, y_train)
	pred = dtree.predict(X_test)
	accuracy = metrics.accuracy_score(y_test, pred)
	print(f"[*] Finished training tree, got accuracy of: {accuracy}%")
	return dtree, accuracy


def train_nn():
	# TODO: This doesn't work. Ensure seed in an other way.
	tensorflow.random.set_seed(tree_seed)
	input_shape = (len(X_train.columns),)

	model = keras.Sequential()
	model.add(Dense(8, input_shape=input_shape, activation="relu"))
	# model.add(Dense(6, input_shape=input_shape, activation="relu"))
	model.add(Dense(2, activation="softmax"))

	model.compile(optimizer=Adam(learning_rate=0.001),
	              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

	print("[*] Started training network")
	history = model.fit(X_train, y_train,
	                    validation_data=(X_test, y_test),
	                    epochs=200, verbose=0, batch_size=128)

	loss, accuracy = model.evaluate(
		X_test, y_test, verbose=0)
	print(f"[*] Finished training net, got accuracy of: {accuracy}%")
	return model, accuracy


classifier, accuracy = train_nn()



@app.get("/onboarding")
@app.get("/")
def onboarding_get():
	return render_template("index.html")


@app.get("/question")
def question_get():
	return render_template("question-page.html",
	                       all_questions=all_questions,
	                       **{fun.__name__: fun for fun in [enumerate, len, time]})


@app.get("/result")
def result_get():
	question_params = {all_questions[int(arg.replace("question-", ""))]: int(val)
	                   for arg, val in request.args.items()
	                   if re.search("^question-\d+$", arg)}

	X_questions = [list(question_params.values())]


	y_proba  = classifier.predict(X_questions)
	_class = numpy.argmax(y_proba, axis=-1)


	divorce_prob = y_proba[0][0]
	print(f"[*] Did a predictionfor X of {X_questions=}: {y_proba=}|{divorce_prob=}")

	return render_template("results-page.html", divorce_prob=divorce_prob,
	                       **{fun.__name__: fun for fun in [round]})


@app.get("/more_info")
def more_info_get():
	return "TODO: Show more info about how the app works", 501


if __name__ == "__main__":
	app.run(debug=0, host="0.0.0.0", port=5000)
