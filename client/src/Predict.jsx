import { useState } from "react";

export default function Predict() {
  const [userInput, setUserInput] = useState();
  const [phrase, setPhrase] = useState();
  const [classification, setClassification] = useState();
  const [certainty, setCertainty] = useState();

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
    ).then((res) => res.json());

    parseInt(prediction.prediction)
      ? setClassification("Positive Sentiment")
      : setClassification("Negative Sentiment");

    if (parseFloat(prediction.output) >= 0.5) {
      let result = ((parseFloat(prediction.output) - 0.5) / 0.5) * 100;
      result = (Math.round(result * 100) / 100).toFixed(2);
      setCertainty(result);
    } else {
      let result = ((0.5 - parseFloat(prediction.output)) / 0.5) * 100;
      result = Math.round(result * 100) / 100;
      setCertainty(result);
    }
  };
  console.log(certainty);

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

      {certainty && (
        <p style={{ fontStyle: "italic" }}>
          AI Model feels {" " + certainty + "% "}certain it classified
          correctly!
        </p>
      )}
    </>
  );
}
