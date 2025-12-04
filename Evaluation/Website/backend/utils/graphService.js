const fs = require("fs");
const path = require("path");
const { runProgram } = require("./pythonService");

/**
 * Converts an array into { "0": value, "1": value, ... }
 */
function arrayToIndexedObject(arr) {
    // If null/undefined → return empty object
    if (arr == null) return {};

    // If already an object with numeric keys → return as-is
    if (typeof arr === "object" && !Array.isArray(arr)) {
        const keys = Object.keys(arr);
        if (keys.every(k => !isNaN(Number(k)))) {
            return arr; 
        }
    }

    // If arr is NOT an array, wrap it
    if (!Array.isArray(arr)) {
        return { 0: arr };
    }

    // Convert array → indexed object
    const obj = {};
    arr.forEach((v, i) => (obj[i] = v));
    return obj;
}

/**
 * Converts cowData into the format required by generateGraph.exe
 */
function convertToExeFormat(cowData) {
    return {
        img: arrayToIndexedObject(cowData.img),
        ID: arrayToIndexedObject(cowData.cowID || cowData.ID),
        DEL: arrayToIndexedObject(cowData.DEL),
        BCS: arrayToIndexedObject(cowData.BCS),
        Semaforo: arrayToIndexedObject(cowData.Semaforo)
    };
}

/**
 * Generates a graph for a given cow using a DATA OBJECT
 * instead of dataset.json.
 * @param {object} cowData - All cow values needed by the EXE
 * @returns {Promise<string>} Base64 data URI
 */
async function generateCowGraph(cowData) {
    const tempInput = path.join(__dirname, "..", "..", "temp_dataset.json");
    const output = path.join(__dirname, "..", "..", "output.png");

    // Convert to EXE-compatible format
    const exeJson = convertToExeFormat(cowData);
    const newJson = JSON.stringify(exeJson, null, 2);

    let shouldWrite = true;

    // Skip writing if JSON is identical
    if (fs.existsSync(tempInput)) {
        const existingJson = fs.readFileSync(tempInput, "utf8");
        if (existingJson.trim() === newJson.trim()) {
            shouldWrite = false;
            console.log("✔ temp_dataset.json unchanged — skip write");
        }
    }

    // Only write if changed
    if (shouldWrite) {
        fs.writeFileSync(tempInput, newJson);
        console.log("✏ temp_dataset.json updated");
    }

    // Run EXE
    console.log(tempInput, cowData.ID, output);
    await runProgram(tempInput, cowData.ID, output);
    console.log("✔ Graph generation EXE completed");
    // Convert PNG → Base64
    const base64Image = fs.readFileSync(output, { encoding: "base64" });

    return "data:image/png;base64," + base64Image;
}

module.exports = { generateCowGraph };
