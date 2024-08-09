import { NavLink } from "react-router-dom";

export default function Navbar() {
  return (
    <div id="navbar">
      <h2>Sentiment Analysis</h2>
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
