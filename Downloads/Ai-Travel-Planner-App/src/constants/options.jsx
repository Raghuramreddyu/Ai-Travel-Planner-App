export const SelectTravelesList = [
    {
        id:1, 
        title:"JustMe",
        desc:'A sole traveles in exploration',
        people:'1'
    },{
        id:2,
        title:'A couple',
        desc:"Two travelers in tandem",
        people:'2 People'

    },{
        id:3,
        title:'Family',
        desc:"A group of fun loving adv",
        people:'3 to 5 People'
    },
    {
        id:4,
        title:'Friends',
        desc:"A bunch of thrill-seekers",
        people:'5 to 10 People'
    }


]

export const SelectBudgetOptions=[
    {
        id:1,
        title:"Cheap",
        desc:"Stay conscious of costs"
    },
    {
        id:2,
        title:"Moderate",
        desc:"Keep cost on the average side"

    },
    {
        id:3,
        title:'Luxury',
        desc:'Dont worry about the cost'
    }

]

export const AI_PROMPT = 'Generate Travel Plan for Location : {location}, for {total Days} Days for  {traveler} with a {budget} budget, give me Hotels options list with HotelName, Hotel address, Price, hotel image url,geo coordinates, rating, descriptions and suggest itinerary with placeName, Place Details, Place Image Url, Geo Coordinates, ticket Pricing, Time Travel each of the location for {totalDays} days with each day plan with best time to visit in JSON format.'