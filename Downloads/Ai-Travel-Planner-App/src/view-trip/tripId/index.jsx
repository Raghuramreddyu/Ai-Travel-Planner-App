import { getDoc } from "firebase/firestore";
import { doc } from "firebase/firestore";
import { db } from "@/service/firebaseConfig";
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
// import Information from "../components/InfoSection";
import InfoSection from "../components/infosection";
import Hotels from "../components/Hotels";
import PlacesToVisit from "../components/PlacesToVisit";

const Viewtrip = () => {
  const { tripId } = useParams();
  const [trip, setTrip] = useState([]);

  useEffect(() => {
    tripId&&GetTripData();
  }, [tripId]);
  const GetTripData = async () => {
    const docRef = doc(db, "AITrips", tripId);
    const docSnap = await getDoc(docRef);

    if (docSnap.exists()) {
      console.log("document:", getDoc(docRef));
      setTrip(docSnap.data());
    } else {
      console.log("no such document found");
      toast("no trip found!");
    }
  };
  return (
    <div className="p-10 md:px-20 lg:px-44 xl:px-56">
      <InfoSection trip={trip} />
      <Hotels trip={trip} />
      <PlacesToVisit trip={trip}/>
    </div>
  );
};

export default Viewtrip;
