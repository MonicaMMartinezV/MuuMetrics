const express = require('express');
const router = express.Router();
const getCowController = require('../controllers/getCowInfo');
const getCowDataController = require('../controllers/getCows');


// Route 3: Find, download, and process data for a specific cow ID (GET /cow-data?cow_id=...)
router.get("/", getCowDataController.getCowData);

router.get("/cow/:cowId", getCowController.getCowInfo);

module.exports = router;