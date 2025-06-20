import React, { useState, useEffect } from 'react';
import { GetPlaceDetails, PHOTO_REF_URL } from '@/service/GlobalApi';
import { FaCalendarAlt, FaMoneyBillWave } from 'react-icons/fa';

const UserTripCardItem = ({ trip }) => {
    const [photoUrl, setPhotoUrl] = useState();

    const GetPlacePhoto = async () => {
        const data = {
            textQuery: trip?.userSelection?.location?.label
        };
        const result = await GetPlaceDetails(data).then(resp => {
            const photoName = resp.data.places[0]?.photos?.[3]?.name;
            if (photoName) {
                const PhotoUrl = PHOTO_REF_URL.replace('{NAME}', photoName);
                setPhotoUrl(PhotoUrl);
            }
        });
    };

    useEffect(() => {
        if (trip) GetPlacePhoto();
    }, [trip]);

    return (
        <div className="bg-white rounded-xl shadow-lg overflow-hidden w-[250px] mt-10">
            <div className="relative h-[180px]">
                <img src={photoUrl || "/placeholder.jpg"} alt={trip?.userSelection?.location?.label} className="w-full h-full object-cover" />
                <div className="absolute inset-0 bg-black bg-opacity-30 flex items-center justify-center text-white text-lg font-bold">
                    {trip?.userSelection?.location?.label}
                </div>
            </div>
            <div className="p-4 text-gray-700">
                <div className="flex items-center gap-2 text-sm text-gray-500">
                    <FaCalendarAlt className="text-orange-500" />
                    <span>{trip?.userSelection?.noOfDays} Days Trip</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500 mt-2">
                    <FaMoneyBillWave className="text-green-500" />
                    <span>Budget: {trip?.userSelection?.budget}</span>
                </div>
            </div>
        </div>
    );
};

export default UserTripCardItem;
