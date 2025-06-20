import React from 'react'
import { Link } from 'react-router-dom'
import { useState , useEffect } from 'react';
import { GetPlaceDetails } from '@/service/GlobalApi';
import { PHOTO_REF_URL } from '@/service/GlobalApi';

const PlaceCardItem = ({place}) => {
     const [photoUrl , setPhotoUrl] = useState();
     
    
      const GetPlacePhoto = async()=>{
        const data={
          textQuery:place.placeName
        }
        const result=await GetPlaceDetails(data).then(resp=>{
          console.log(resp.data.places[0].photos[3].name)
    
          const PhotoUrl = PHOTO_REF_URL.replace('{NAME}' , resp.data.places[0].photos[3].name)
          setPhotoUrl(PhotoUrl)
        })
      }
        useEffect(()=>{
        place&&GetPlacePhoto()
       } ,[place])
    
    return (
         <Link to={'https://www.google.com/maps/search/?api=1&query='+place.placeName} target='_blank'>
        <div className='border rounded-xl p-3 mt-2 flex gap-5 hover:scale-105 transition-all hover:shadow-md cursor-pointer'>
        <img src={photoUrl} className='w-[130px] h-[130px] rounded-xl' alt={place.placeName} />
        <div>
            <h2 className='font-bold text-lg'>{place.placeName}</h2>
            <p className='text-sm text-gray-400'>{place.placeDetails}</p>
            <h2 className='mt-2 text-sm text-gray-600'>Time to Travel: {place.timeToTravel}</h2>
            <h2 className='text-sm text-gray-600'>Best Time to Visit: {place.bestTimeToVisit}</h2>
            <h2 className='text-sm text-gray-600'>Opening Hours: {place.openingHours}</h2>
            <h2 className='text-sm text-gray-600'>Ticket Pricing: {place.ticketPricing}</h2>
        </div>
    </div>
    </Link>
    )
}

export default PlaceCardItem
