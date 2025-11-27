const fs = require("fs");
const path = require("path");
const { runProgram } = require("./pythonService");

/**
 * Generates a graph for a given cow ID and returns it as a Base64 data URI.
 * @param {string} cowId
 * @returns {Promise<string>} Base64 data URI of the generated PNG
 */
async function generateCowGraph(cowId) {
    // 1️⃣ Paths
    const dataset = path.join(__dirname, "..", "..", "dataset.json");
    const output = path.join(__dirname, "..", "..", "output.png"); // you can also name it cow_${cowId}.png

    // 2️⃣ Run the EXE
    await runProgram(dataset, cowId, output);

    // 3️⃣ Read the PNG and convert to Base64
    const base64Image = fs.readFileSync(output, { encoding: "base64" });
    const dataUri = "data:image/png;base64," + base64Image;

    return dataUri;
}

module.exports = { generateCowGraph };
