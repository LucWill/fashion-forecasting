
from fasthtml.common import (fast_app, serve, Form, Input, Textarea, Select,
                             Button, Div, Label, Option, H1, HTMLResponse)
import pandas as pd
from fashion_pipeline import load_model
from spacy.tokens import Doc
from spacytextblob.spacytextblob import SpacyTextBlob  # noqa: F401


# Create the main app and routing
app, route = fast_app()


@route('/')
def home():
    form = Form(
        Label("Review Title"),
        Input(name="title", type="text"),

        Label("Review Text"),
        Textarea(name="review_text"),

        Label("Customer Age"),
        Input(name="age", type="number"),

        Label("Division"),
        Select(
            Option("General", value="General"),
            Option("General Petite", value="General Petite"),
            name="division"
        ),

        Label("Department"),
        Select(
            Option("Tops", value="Tops"),
            Option("Dresses", value="Dresses"),
            Option("Bottoms", value="Bottoms"),
            Option("Intimate", value="Intimate"),
            Option("Jackets", value="Jackets"),
            Option("Trend", value="Trend"),
            name="department"
        ),

        Label("Class"),
        Select(
            Option("Dresses", value="Dresses"),
            Option("Knits", value="Knits"),
            Option("Blouses", value="Blouses"),
            Option("Sweaters", value="Sweaters"),
            Option("Pants", value="Pants"),
            Option("Jeans", value="Jeans"),
            Option("Fine gauge", value="Fine gauge"),
            Option("Skirts", value="Skirts"),
            Option("Jackets", value="Jackets"),
            Option("Outerwear", value="Outerwear"),
            Option("Shorts", value="Shorts"),
            Option("Lounge", value="Lounge"),
            Option("Trend", value="Trend"),
            Option("Casual bottoms", value="Casual bottoms"),
            name="class_name"
        ),

        Button("Predict", type="submit"),
        hx_post="/predict",
        hx_target="#prediction-output",
        id="review-form"
    )

    return Div(
        H1("StyleSense Recommendation Predictor"),
        form,
        Div(id="prediction-output")
    )


@app.post('/predict')
async def predict(r):
    form = await r.form()
    df = pd.DataFrame([{
        "Title": form["title"],
        "Review Text": form["review_text"],
        "Age": float(form["age"]),
        "Division Name": form["division"],
        "Department Name": form["department"],
        "Class Name": form["class_name"]
    }])
    if not Doc.has_extension("blob"):
        Doc.set_extension("blob", default=None)
    model = load_model()
    prob = model.predict_proba(df)[0][1]
    response_html = f"""<div><strong>Recommendation Probability:
        </strong> {prob:.2f}</div>"""
    return HTMLResponse(response_html)

# Start the server
if __name__ == "__main__":
    serve()
