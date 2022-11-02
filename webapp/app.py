# Webapp imports
import random
import string
import threading
from pprint import pprint
import time
import re
import os


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
seed = 46841
overwrite_model = False

all_questions = [
	"My spouse and I have similar ideas about how roles should be in marriage",
	"I enjoy traveling with my wife.",
	"The time I spent with my wife is special for us.",
	"I know my spouse's basic anxieties.",
	"My discussion with my spouse is not calm.",
	"I know my spouse's favorite food.",
	"I know my spouse's friends and their social relationships.",
	"I know what my spouse's current sources of stress are.",
]

"""
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
"""



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
	os.getenv("DATASET_FILE") or ".\\dataset\\divorce.xlsx",
	os.getenv("DESCRIPTION_FILE") or ".\\dataset\\reference.tsv")


def train_tree():
	print(f"[*] Creating descision tree with seed: {seed}")
	dtree = DecisionTreeClassifier(random_state=seed, max_depth=5)
	_tree_hist = dtree.fit(X_train, y_train)
	pred = dtree.predict(X_test)
	accuracy = metrics.accuracy_score(y_test, pred)
	print(f"[*] Finished training tree, got accuracy of: {accuracy}%")
	return dtree, accuracy


def train_nn(save_model_path="./models/default.h5", restore_model_path="./models/default.h5"):
	print(f"[*] Random: {''.join([str(list(tensorflow.random.normal((1, 1)).numpy()[0])[0])[-1] for x in range(10)])}")
	if not os.path.isfile(save_model_path) or overwrite_model:
		input_shape = (len(X_train.columns),)
		model = keras.Sequential()
		model.add(Dense(8, input_shape=input_shape, activation="relu"))
		model.add(Dense(6, input_shape=input_shape, activation="relu"))
		# model.add(Dense(4, input_shape=input_shape, activation="relu"))
		model.add(Dense(2, activation="softmax"))

		model.compile(optimizer=Adam(learning_rate=0.001),
		              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

		print("[*] Started training network")
		history = model.fit(X_train, y_train,
		                    validation_data=(X_test, y_test),
		                    epochs=200, verbose=0, batch_size=128)

		model.save(save_model_path)
	else:
		print(f"[*] Restored network: {restore_model_path}")
		model = keras.models.load_model(restore_model_path)

	loss, accuracy = model.evaluate(
		X_test, y_test, verbose=0)
	print(f"[*] Finished training net, got accuracy of: {accuracy}%")
	return model, accuracy

classifier, accuracy = train_nn(
	 os.getenv("SAVE_MODEL_PATH") or ".\\models\\default.h5",
	 os.getenv("RESTORE_MODEL_PATH") or ".\\models\\default.h5"
)


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

	y_proba = classifier.predict(X_questions)
	_class = numpy.argmax(y_proba, axis=-1)

	divorce_prob = y_proba[0][0]
	print(f"[*] Did a prediction for X of {X_questions=}: {y_proba=}|{divorce_prob=}")

	if divorce_prob < .33:
		chance = "low"
		percentage_color = "green"
		improvement_text = "You have a " + chance + " chance of getting a divorce. But this does not mean "

		tips_title = "Here are some tips to make your marriage even better!"
		tips_tagline = "Although your chances to get a divorce are quite low, there's awlays room " \
		               "for improvement. Here are some tips to make your marriage even better. "
		tips = [
			[
				"Have fun together",
				"""Having fun together is what makes relationships great. 
				Always make sure to have fun together. Make time to do things together, 
				go out to eat, and just enjoy each other's company."""
			],
			[
				"Give compliments",
				"""
				Complimenting someone is a great way to show them 
				how much you care about them. Complimenting someone 
				shows them that you notice their positive traits and 
				appreciate them.
				"""
			],
			[
				"Listen",
				"""
				Listening is a skill that everyone should learn. 
				Listening helps you understand what someone else is 
				saying. It helps you connect with them and build trust 
				between you. So if you want a good relationship, listen!
				"""
			]
		]

	elif divorce_prob > .33 and divorce_prob < .66:
		chance = "medium"
		percentage_color = "orange"
		tips_title = "Here are some tips to improve your mariage."
		tips_tagline = ""
		tips = [
			[
				"Respect each other's space",
				"""Respecting each others' personal space is something
				   that should always be done. You should never invade someone else's personal 
				   space without their permission. When you respect 
				   each other's space, you're showing them that they
				  matter to you."""
			],
			[
				"Be honest about what you want",
				"""The first step to preventing a divorce is being 
					honest about what you want. If you don't know what you want, 
					then how do you expect someone else to know? You need to have 
					a clear idea of what you want out of a relationship before you 
					enter into one. Don't get married just because everyone else does. 
					Make sure you're ready to commit to each other."""
			],
			[
				"Keep communication open",
				"""
				Communication is key to any relationship. 
				Whether it's with family members, friends, 
				or even lovers, keeping communication open 
				is always a good idea. When you talk to each other, 
				you get to know what makes them tick. 
				And when you know what makes them tick, 
				you can help them feel comfortable around you.
				"""
			]
		]
	elif divorce_prob > .66:
		chance = "high"
		percentage_color = "red"
		tips_title = "Here are some tips to work on your mariage"
		tips_tagline = "The results of our analysis on your data indicate that your have a high chance" \
		               "to get divorced. Bellow are some tips to imrpove your relationship. In adition to that we" \
		               "would highly recommend to contact a relationship professional."
		tips = [
			[
				"Communication",
				"""Communication is the first step to any relationship. 
				If you want to keep things going, communication is key. 
				You should always make sure to communicate openly and honestly 
				about what's going on in your lives."""
			],
			[
				"Accept responsibility",
				"""
				A lot of people blame others for problems in 
				relationships, but the real problem lies with 
				the individual. You have to accept responsibility 
				for your actions. If you're constantly blaming others, 
				then you're going to keep having problems.
				"""

			],
			[
				"Respect each other",
				"""
				Respect is something that comes naturally 
				between two people who love each other. 
				When you respect someone, you show them that 
				you care about them and value their opinion. 
				Showing someone you respect them makes them 
				feel good about themselves.
				"""
			]
		]

	return render_template("results-page.html",
	                       percentage_color=percentage_color,
	                       divorce_prob=divorce_prob,
	                       tips_tagline=tips_tagline,
	                       tips_title=tips_title,
	                       tips=tips,
	                       **{fun.__name__: fun for fun in [round]})


@app.get("/more_info")
def more_info_get():
	return "TODO: Show more info about how the app works", 501


if __name__ == "__main__":
	app.run(debug=1, host="0.0.0.0", port=os.getenv("PORT") or 5000)
