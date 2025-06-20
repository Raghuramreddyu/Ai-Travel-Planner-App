import React , {useEffect , uesState, useState} from 'react'
import { db } from '@/service/firebaseConfig'
import { collection,query , where } from 'firebase/firestore'
import { useNavigation } from 'react-router-dom'
import UserTripCardItem from './UserTripCardItem'
import { getDocs } from 'firebase/firestore'
const MyTrips = () => {
    const [userTrips , setuserTrips] = useState([])
   useEffect(()=>{
    GetUserTrips()
   } , [])
   const navigation = useNavigation()
   const GetUserTrips= async()=>{
    const user= JSON.parse(localStorage.getItem('user'))
   
   if(!user){
 

        navigation('/');
        return;
    }
    setuserTrips([])
    const q =query(collection(db, 'AITrips'),where('userEmail', '==', user?.email))
    const querySnapshot = await getDocs(q);
     querySnapshot.forEach((doc) => {
 
     console.log(doc.id, " => ", doc.data());
     setuserTrips(prev=>[...prev, doc.data()])
});  
   }




    return (
        <div className='sm:px-10 md:px-32 lg:px-56 xl:px-72 px-5 mt-10'>
         <h2 className='font-bold text-3xl' >My trips</h2> 

         <div className='grid grid-cols-2 mt-10 md:grid-cols-73'>
            {userTrips.map((trip , index)=>(
                <UserTripCardItem trip={trip}
                 className="object-cover rounded-xl" />
            ))}
         </div>
        </div>
    )
}

export default MyTrips
