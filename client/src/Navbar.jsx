import { NavLink } from "react-router-dom";

export default function Navbar() {
  return (
    <div id="navbar">
      <h1>Sentiment Analysis</h1>
      <nav className="navbar-links">
        <NavLink
          to="/"
          className={({ isActive }) =>
            isActive ? "navbar-link active" : "navbar-link"
          }
        >
          Home
        </NavLink>
        <NavLink
          to="/about"
          className={({ isActive }) =>
            isActive ? "navbar-link active" : "navbar-link"
          }
        >
          About
        </NavLink>
      </nav>
    </div>
  );
}
