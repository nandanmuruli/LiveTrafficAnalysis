const mongoose = require("mongoose");

const trafficStatsSchema = new mongoose.Schema(
  {
    timebucket: { type: Date, required: true },
    type: {
      type: String,
      enum: ["minute", "hourly", "daily", "weekly"],
      required: true,
    },

    vehicleCount: { type: Number, default: 0 },
    totalTravelTime: { type: Number, default: 0 },
    counts: {
      type: Map,
      of: Number,
      default: {},
    },
totalViolations: { type: Number, default: 0 },
  },
  { timestamps: true }
);

trafficStatsSchema.index({ timebucket: 1, type: 1 }, { unique: true });

module.exports = mongoose.model("TrafficStats", trafficStatsSchema);
