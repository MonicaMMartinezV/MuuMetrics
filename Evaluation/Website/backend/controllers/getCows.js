const cowInfo = require('../model/model');

exports.refreshCowList = async (req, res) => {
    try {
        const cows = await cowInfo.findAll();

        res.render("getCows", {
            cows
        });
    } catch (error) {
        console.error(error);
        res.status(500).send("Error refreshing cow list");
    }
};