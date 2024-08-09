import "./App.css";
import Navbar from "./Navbar";
import { Outlet } from "react-router-dom";

function App() {
  return (
    <>
      <Navbar />
      <p style={{ fontStyle: "italic", marginBottom: "8px" }}>
        "A very limited and biased Sentiment analysis AI model trained on 1.6
        Million Tweets err... X posts"
      </p>
      <Outlet />
    </>
  );
}

export default App;
