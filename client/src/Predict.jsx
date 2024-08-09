import { useState } from "react";

export default function Predict() {
  const [userInput, setUserInput] = useState();
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
        <h3>
          The phrase {userInput} is classified as:{classification}
        </h3>
      )}
    </>
  );
}
