import { GetPlaceDetails } from '@/service/GlobalApi'
import React, { useEffect ,useState } from 'react'
import { PHOTO_REF_URL } from '@/service/GlobalApi';

const InfoSection = ({trip}) => {
  const [photoUrl , setPhotoUrl] = useState();
 

  const GetPlacePhoto = async()=>{
    const data={
      textQuery:trip?.userSelection?.location?.label
    }
    const result=await GetPlaceDetails(data).then(resp=>{
      console.log(resp.data.places[0].photos[3].name)

      const PhotoUrl = PHOTO_REF_URL.replace('{NAME}' , resp.data.places[0].photos[3].name)
      setPhotoUrl(PhotoUrl)
    })
  }
    useEffect(()=>{
    trip&&GetPlacePhoto()
   } ,[trip])

  return (
    <div>
      <img src={photoUrl} className='h-[300px] w-full object-cover rounded-xl'/>
    <div>
      <div className='my-5 flex flex-col gap-2'>
        <h2 className='font-bold text-xl'>{trip?.userSelection?.location?.label}</h2>
     <div className='flex gap-5'>
     <h2 className='p-1 px-3 bg-gray-200 rounded-full  text-gray-500'>📅{trip.userSelection?.noOfDays}</h2>
      <h2 className='p-1 px-3 bg-gray-200 rounded-full  text-gray-500'>💰{trip.userSelection?.budget}</h2>
      <h2 className='p-1 px-3 bg-gray-200 rounded-full  text-gray-500'>🥂No of Travelers: {trip.userSelection?.traveler}</h2>
     </div>
      
     
      </div>
    </div>
  </div>
  )
}

export default InfoSection
