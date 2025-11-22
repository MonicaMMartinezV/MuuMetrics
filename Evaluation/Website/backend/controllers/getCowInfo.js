const cowInfo = require('../model/model');

exports.getCowInfo = async (req, res) => {
    const cowId = req.params.id;
    try {
        const cow = await cowInfo.findOne({ where: { IDCow: cowId } });

        // const healthStatus = await getCowHealthStatus(cow);
        const healthStatus = { status: "rojo" };

        res.render("getCowInfo", {
            cowID: cow.IDCow,
            bcs: cow.BCS,
            diasLeche: cow.DEL,
            estado: "Temporal",
            status: healthStatus.status
        });
    } catch (error) {
        res.status(500).send('Error retrieving cow information');
    }
};