import { useState } from "react";

export default function Predict() {
  const [userInput, setUserInput] = useState();
  const [phrase, setPhrase] = useState();
  const [classification, setClassification] = useState();

  const handleChange = (e) => {
    setUserInput(e.target.value);
  };

  const analyzeInput = async (e) => {
    e.preventDefault();
    if (!userInput) {
      alert("Please write a phrase!");
      return;
    }
    setPhrase(userInput);
    e.target.reset();

    const prediction = await fetch(
      "https://pytorchsentiment-production.up.railway.app/predict",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: userInput }),
      }
    )
      .then((res) => res.json())
      .then((res) => res.prediction);

    parseInt(prediction)
      ? setClassification("Positive Sentiment")
      : setClassification("Negative Sentiment");
  };

  return (
    <>
      <form id="predict-form" onSubmit={analyzeInput}>
        <div id="input-and-button">
          <input
            id="predict-input"
            placeholder="Write a short sentence to analyze"
            maxLength="280"
            onChange={handleChange}
          />
          <button id="input-button">Analyze Sentiment</button>
        </div>
      </form>

      {classification && (
        <div className="results">
          <h3 style={{ textWrap: "wrap", width: "700px" }}>
            "<span style={{ fontWeight: "bolder" }}>{phrase}</span>"<br></br>
            <br></br> Classified as:{" "}
            <span
              style={{
                color: classification == "Positive Sentiment" ? "green" : "red",
                fontWeight: "bolder",
              }}
            >
              {classification}!
            </span>
          </h3>
          {classification == "Positive Sentiment" ? (
            <p
              style={{
                fontSize: "100px",
                margin: "0px",
              }}
              className="emoji"
            >
              &#128513;
            </p>
          ) : (
            <p
              style={{
                fontSize: "100px",
                margin: "0px",
              }}
              className="emoji"
            >
              &#128530;
            </p>
          )}
        </div>
      )}
    </>
  );
}
