const express = require("express");
const path = require("path");
1
const app = express();

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

app.use(express.static(path.join(__dirname, "public")));

app.get("/", getCows);
app.get("/getCows", getCows);
app.get("/getCowInfo/:id", getCowInfo);


app.listen(3000, () => {
    console.log("Servidor corriendo en http://localhost:3000");
});
