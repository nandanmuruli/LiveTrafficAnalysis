const path = require("path");
const fs = require("fs");

const VehicleEvent = require("../models/VehicleEvent");
const { getCurrentWeather } = require("./weatherService");
const { embedText } = require("./embeddingService");
const { buildSentence } = require("../utils/sentanceBuilder");
const { upsertVehicle } = require("./qdrantService");
const TrafficStats = require("../models/TrafficStats");

const IMG_DIR = path.resolve(__dirname, "../../assets/img");

if (!fs.existsSync(IMG_DIR)) {
	fs.mkdirSync(IMG_DIR, { recursive: true });
}

function saveImageToDisk(imageBuffer, vehicleId) {
	if (!imageBuffer) return null;

	try {
		const filename = `vehicle_${vehicleId}_${Date.now()}.jpg`;
		const filepath = path.join(IMG_DIR, filename);
		fs.writeFileSync(filepath, imageBuffer);
		return `assets/img/${filename}`;
	} catch (err) {
		console.error("[MongoService] Failed to save image to disk:", err.message);
		return null;
	}
}

async function saveEvent(eventData, imageBuffer, imageVector) {
	try {
		const imagePath = saveImageToDisk(imageBuffer, eventData.vehicle_id);

		const event = new VehicleEvent({
			...eventData,
			...(imagePath && { image_path: imagePath }),
		});
		await event.save();
		console.log(`[MongoService] Saved ${event.class} ID:${event.vehicle_id}`);

		const weather = await getCurrentWeather();
		const sentence = buildSentence(event, weather);
		const textVector = await embedText(sentence);

		await upsertVehicle(event, sentence, textVector, imageVector, weather);

		await updateAggregates(eventData);
	} catch (error) {
		console.error("[MongoService] Failed to save event:", error.message);
	}
}

async function updateAggregates(eventData) {
	try {
		const calculateDuration = (ent, ext) => {
			if (!ent || !ext) return 0;
			const start = new Date(`1970-01-01T${ent}Z`);
			const end = new Date(`1970-01-01T${ext}Z`);
			const diff = (end - start) / 1000;
			if (diff > 0 && diff < 5) {
				diff += 15.0;
			}
			return diff > 0 ? diff : 0;
		};
		const travelDuration = calculateDuration(
			eventData.entry_time,
			eventData.exit_time,
		);
		const ts =
			eventData.timestamp < 10000000000
				? eventData.timestamp * 1000
				: eventData.timestamp;
		const date = new Date(ts);
		const minuteBucket = new Date(date);
		minuteBucket.setSeconds(0, 0);

		const hourBucket = new Date(date);
		hourBucket.setMinutes(0, 0, 0);

		const dayBucket = new Date(date);
		dayBucket.setHours(0, 0, 0, 0);

		const weekBucket = getStartOfWeek(date);

		const buckets = [
			{ bucket: minuteBucket, type: "minute" },
			{ bucket: hourBucket, type: "hourly" },
			{ bucket: dayBucket, type: "daily" },
			{ bucket: weekBucket, type: "weekly" },
		];

		for (const item of buckets) {
			await TrafficStats.findOneAndUpdate(
				{ timebucket: item.bucket, type: item.type },
				{
					$inc: {
						[`counts.${eventData.class}`]: 1,
						vehicleCount: 1,
						totalTravelTime: travelDuration,
						totalViolations: eventData.isViolation ? 1 : 0,
					},
				},
				{ upsert: true },
			);
			console.log(`[AGGREGATE] Updated all time-scales for ${eventData.class}`);
		}
	} catch (error) {
		console.error("Aggregation Error:", error.message);
	}
}
function getStartOfWeek(d) {
	const date = new Date(d);
	const day = date.getDay();
	const diff = date.getDate() - day + (day === 0 ? -6 : 1);
	return new Date(date.setDate(diff)).setHours(0, 0, 0, 0);
}

module.exports = { saveEvent };
