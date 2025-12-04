const express = require('express');
const router = express.Router();
const getCowController = require('../controllers/getCowInfo');
const getCowDataController = require('../controllers/getCows');

// Main homepage route
router.get("/", getCowDataController.getCowData);

// NEW: Fast loader route
router.get("/cow/:cowId", getCowController.showCowLoader);

// Slow heavy route
router.get("/cow/:cowId/info", getCowController.getCowInfo);

module.exports = router;
