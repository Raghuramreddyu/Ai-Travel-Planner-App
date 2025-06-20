import { StrictMode   } from 'react'
import React from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { RouterProvider , createBrowserRouter } from 'react-router-dom'
// import createtrip from './create-trip/create-trip.jsx'
import Createtrip from './create-trip/index.jsx'
import Header from './components/custom/Header'
import { Toaster } from 'sonner'
import { GoogleOAuthProvider } from '@react-oauth/google'
import Viewtrip from './view-trip/tripId/index.jsx'
import MyTrips from './my-trips/index.jsx'
const router = createBrowserRouter([
  {
    path: '/', 
    element:<App/>
  },
  {
    path:'/create-trip',
    element:<Createtrip/>

  },

  {
    path:'/view-trip/:tripId',
    element:<Viewtrip/>
  },{
    path:'/mytrips',
    element:<MyTrips/>
  }
])

createRoot(document.getElementById('root')).render(
<GoogleOAuthProvider clientId={import.meta.env.VITE_GOOGLE_AUTH_CLIENT_ID}>
  <React.StrictMode>
    <Header/>
    <Toaster />
    <RouterProvider router = {router} />
  </React.StrictMode>
  </GoogleOAuthProvider>

)
