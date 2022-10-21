import random
import string

from flask import Flask, render_template, request, redirect


app = Flask(__name__)
admin_password = "admin"

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


@app.get("/train")
def train():
	# Probably very insecure auth, but no problem for this small scale project.
	password = request.args.get("password", default="", type=str)
	storage = request.args.get("storage", default="mem", type=str)
	if password == admin_password:
		if storage == "mem":
			tree = None
		elif storage == "disk":
			tree = None
			# tree.save or somethign
		else:
			return "Storage should either be disk or mem", 500
		return "TODO: Train the tree here and save it to disk|mem", 501
	else:
		return "Unauthenticated", 401


@app.get("/question")
def question_get():
	question_id = request.args.get("question_id", default=0, type=int)
	# Not truely super duper random, but same as above
	_session_id_code = "".join(
		random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
	session_id = request.args.get("session_id", type=str, default=_session_id_code)
	if question_id >= len(all_questions):
		return "You're finished. Your result is: " + session_id

	question = all_questions[question_id]
	print(f"New req: {question_id}")
	return render_template("question-page.html",
	                       question=question,
	                       question_id=question_id,
	                       all_questions=all_questions,
	                       session_id=session_id,
	                       len=len)

@app.get("/onboarding")
@app.get("/")
def onboarding_get():
	return "TODO: Onboarding", 501

@app.get("/result")
def result_get():
	return "TODO: Show results page", 501

@app.get("/more_info")
def more_info_get():
	return "TODO: Show more info about how the app works", 501

if __name__ == "__main__":
	app.run(debug=True, host="0.0.0.0", port=5000)
