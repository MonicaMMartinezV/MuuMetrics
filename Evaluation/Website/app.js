const express = require("express");
const path = require("path");
const cowRoutes = require('./backend/routes/cowRoutes.js'); // Import the router

const app = express();

// --- View Engine Setup (EJS) ---
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "frontend", "views"));


app.use(express.static(path.join(__dirname, "frontend", "public")));

// --- Routes ---

// Use the existing cowRoutes for API endpoints and file listing
app.use("/", cowRoutes);


app.listen(3000, () => {
    console.log("Servidor corriendo en http://localhost:3000");
});