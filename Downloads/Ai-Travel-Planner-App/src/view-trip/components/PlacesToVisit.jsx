import React from "react";
import PlaceCardItem from "./PlaceCardItem";

const PlacesToVisit = ({ trip }) => {
  return (
    <div>
      <h2 className="font-bold text-lg">Places to Visit</h2>
      <div>
        {trip?.tripData?.itinerary &&
        typeof trip.tripData.itinerary === "object" ? (
          Object.entries(trip.tripData.itinerary).map(
            ([day, details], index) => (
              <div key={index}>
                <h2 className="font-medium text-lg">
                  Day {index + 1}: {details.theme}
                </h2>
                <div className="grid md:grid-cols-2 gap-5">
                  {Object.entries(details).map(
                    ([timeOfDay, place], idx) =>
                      timeOfDay !== "theme" &&
                      place && ( // Ensure place is not null/undefined
                        <div key={idx} className="">
                          <h2 className="font-medium text-sm text-orange-600">
                            {timeOfDay.charAt(0).toUpperCase() +
                              timeOfDay.slice(1)}
                          </h2>
                          <PlaceCardItem place={place} />
                        </div>
                      )
                  )}
                </div>
              </div>
            )
          )
        ) : (
          <p>No itinerary available.</p> // Handle empty state gracefully
        )}
      </div>
    </div>
  );
};

export default PlacesToVisit;
