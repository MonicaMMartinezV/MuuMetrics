const cowInfo = require('../models/Cow');

exports.getCows = async (req, res) => {
    try {
        const cows = await cowInfo.findAll();

        res.render("cows", {
            cows
        });
    } catch (error) {
        console.error(error);
        res.status(500).send("Error refreshing cow list");
    }
};