# Webapp imports
import random
import string
import threading
import time
import re

from flask import Flask, render_template, request, redirect

# Tree imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from random import randint

app = Flask(__name__)
tree, accuracy = None, None
admin_password = "admin"
tree_seed = 46841

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


def prepare_dataset(dataset_file="./dataset/divorce.xlsx",
                    description_file="./dataset/refercence.tsv"):
	headers_frame = pd.read_csv(description_file, sep="|")
	headers = list(headers_frame["description"])
	headers.append("Class")
	df = pd.read_excel(dataset_file, names=headers)

	X, y = df[all_questions], df["Class"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	return df, X_train, X_test, y_train, y_test


def train_tree():
	print("[*] Preparing dataset")
	df, X_train, X_test, y_train, y_test = prepare_dataset(
		".\\dataset\\divorce.xlsx", ".\\dataset\\reference.tsv")

	print(f"[*] Creating descision tree with seed: {tree_seed}")
	dtree = DecisionTreeClassifier(random_state=tree_seed, max_depth=5)
	_tree_hist = dtree.fit(X_train, y_train)
	pred = dtree.predict(X_test)
	accuracy = metrics.accuracy_score(y_test, pred)
	print(f"[*] Finished training tree, got accuracy of: {accuracy}%")

	return dtree, accuracy


@app.before_first_request
def activate_job():
	global tree, accuracy
	tree, accuracy = train_tree()


@app.get("/onboarding")
@app.get("/")
def onboarding_get():
	return render_template("index.html")


@app.get("/question")
def question_get():
	question_id = request.args.get("question_id", default=0, type=int)
	# Not truely super duper random, but same as above
	_session_id_code = "".join(
		random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
	session_id = request.args.get("session_id", type=str, default=_session_id_code)

	print(f"New req: {question_id}")
	return render_template("question-page.html",
	                       all_questions=all_questions,
	                       session_id=session_id,
	                       **{fun.__name__: fun for fun in [enumerate, len, time]})


@app.get("/result")
def result_get():
	question_params = {all_questions[int(arg.replace("question-", ""))]: int(val)
	                   for arg, val in request.args.items()
	                   if re.search("^question-\d+$", arg)}

	X_questions = [list(question_params.values())]
	pred, prob = tree.predict_proba(X_questions)[0]
	return render_template("results-page.html", pred=pred, prob=prob)


@app.get("/more_info")
def more_info_get():
	return "TODO: Show more info about how the app works", 501


if __name__ == "__main__":
	app.run(debug=True, host="0.0.0.0", port=5000)
