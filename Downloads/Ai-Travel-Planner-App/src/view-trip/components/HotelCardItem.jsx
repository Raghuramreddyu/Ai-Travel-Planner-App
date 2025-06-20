import React from 'react'
import { Link } from 'react-router-dom'
import { PHOTO_REF_URL } from '@/service/GlobalApi';
import { useState , useEffect } from 'react';
import { GetPlaceDetails } from '@/service/GlobalApi';
const HotelCardItem = ({hotel}) => {

     const [photoUrl , setPhotoUrl] = useState();
     
    
      const GetPlacePhoto = async()=>{
        const data={
          textQuery:hotel?.hotelName
        }
        const result=await GetPlaceDetails(data).then(resp=>{
          console.log(resp.data.places[0].photos[3].name)
    
          const PhotoUrl = PHOTO_REF_URL.replace('{NAME}' , resp.data.places[0].photos[3].name)
          setPhotoUrl(PhotoUrl)
        })
      }
        useEffect(()=>{
        hotel&&GetPlacePhoto()
       } ,[hotel])
       
    return (
        <div>
      < Link to={'https://www.google.com/maps/search/?api=1&query='+hotel.hotelName+","+hotel?.hotelAddress } target='_blank'>
                    <div className='hover:scale-110 transition-all'>
                        <img src={photoUrl} className='rounded-xl h-[200px] w-[200px]' /> 
                        <div className='flex flex-col my-2 gap-2'>
                            <h2 className='font-medium'>{hotel?.hotelName}</h2>
                            <h2 className='text-xs text-gray-500'>📍{hotel?.hotelAddress}</h2> 
                            <h2 className='text-sm'>💰{hotel?.pricePerNight}</h2>     
                            <h2 className='text-sm'>⭐{hotel?.rating}</h2>                          
                        </div>
                    </div>
                    </Link>
        </div>
    )
}

export default HotelCardItem
