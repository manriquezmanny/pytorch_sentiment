export default function About() {
  return (
    <>
      <h1>
        This fullstack app was developed by Manuel Manriquez who trained the AI
        Model using PyTorch
      </h1>
      <ul style={{ textAlign: "left" }}>
        <li>
          <h2 style={{ marginBottom: "0px", textAlign: "left" }}>
            What I learned from this project:
          </h2>
          <ol>
            <li>Finding suitable data</li>
            <li>Pre-processing/cleaning the data</li>
            <li>Encoding the data so it could be fed to a Neural Network</li>
            <li>Craeting a vocab map of tokenized data using nltk</li>
            <li>
              Creating a Neural Network using Convolutional Neural Network
              Layers
            </li>
            <li>Max Pooling and Dropout layers</li>
            <li>Early Stopping condition based on validation loss</li>
            <li>Tuning Hyperparameters to increase accuracy and reduce loss</li>
            <li>Hosting my model and vocab using HuggingFace CLI</li>
            <li>
              Creating a Flask API that can load and use the model and vocab
              using HuggingFace CLI
            </li>
            <li>Hosting the Flask API on Railway</li>
            <li>
              Created a frontend React GUI that connects to backend which loads
              the model
            </li>
          </ol>
          <br></br>
          <h2 style={{ marginBottom: "0px", textAlign: "left" }}>
            Fun facts about the project:
          </h2>
          <li>
            The dataset I used:{" "}
            <a
              style={{ textDecoration: "underline" }}
              href="https://www.kaggle.com/datasets/kazanova/sentiment140"
            >
              Sentiment140
            </a>
          </li>
          <li>
            This AI Model reached a validation accuracy of 82% which is good
            considering humans tend to agree upon sentiment at around 80-85%
            according to various studies.
          </li>
          <li>
            Over 500 Million tweets are sent out daily... my model reached this
            level of performance with only 1.6 Million. Now imagine Large Models
            that have access to the whole internet and enormous processing
            power!
          </li>
          <li>
            This AI model, like many others, is biased because it was trained on
            Twitter posts and twitter has collective biases.
          </li>
          <li>
            I hosted my AI model on{" "}
            <a
              style={{ textDecoration: "underline" }}
              href="https://huggingface.co/manriquezmanny/pytorch-sentiment/tree/main"
            >
              HuggingFace
            </a>
          </li>
          <li>
            This was a supervised learning project since the data I used was
            labeled
          </li>
        </li>
      </ul>
    </>
  );
}
