import React from "react";
import * as ReactDOM from "react-dom/client";
import App from "./App.jsx";
import Predict from "./Predict.jsx";
import About from "./About.jsx";
import "./index.css";
import {
  createBrowserRouter,
  RouterProvider,
  Navigate,
} from "react-router-dom";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { element: <Navigate to="/" replace /> },
      { path: "/", element: <Predict /> },
      { path: "/about", element: <About /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
