const express = require("express");
const path = require("path");
const app = express();

const { getCows } = require("./backend/controllers/getCows");
const { getCowInfo } = require("./backend/controllers/getCowInfo");

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

app.use(express.static(path.join(__dirname, "public")));

/*app.get("/", getCows);*/
app.get("/getCows", getCows);
app.get("/getCowInfo/:id", getCowInfo);

app.get("/", (req, res) => {
    res.redirect("/getCows");
});

app.listen(3000, () => {
    console.log("Servidor corriendo en http://localhost:3000");
});
