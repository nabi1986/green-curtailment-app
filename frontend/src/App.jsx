import { useEffect, useState } from 'react'
import { getForecast, postBid } from './api'
import KPICards from './components/KPICards'
import ForecastChart from './components/ForecastChart'
import ChatBox from './components/ChatBox'

export default function App() {
  const [data, setData] = useState(null)
  const [bid, setBid] = useState(null)

  useEffect(() => {
    getForecast("site-A", 24).then(setData)
  }, [])

  const makeBid = async () => {
    const res = await postBid({ site_id: "site-A", horizon_hours:24, features:{} })
    setBid(res)
  }

  const latest = data?.points?.[0]

  return (
    <div style={{maxWidth:1200, margin:'0 auto', padding:16, fontFamily:'system-ui, sans-serif'}}>
      <h2>Energy Curtailment Dashboard</h2>
      <KPICards latest={latest} />
      <ForecastChart data={data?.points||[]} />
      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginTop:16}}>
        <div style={{border:'1px solid #eee', borderRadius:8, padding:12}}>
          <h3>Bid Simulator</h3>
          <button onClick={makeBid} style={{padding:'8px 12px', borderRadius:6, border:'1px solid #ccc'}}>Compute Bid</button>
          {bid && (
            <div style={{marginTop:12}}>
              Quantity: <b>{bid.qty_mw} MW</b> | Price: <b>{bid.price_eur_mwh} €/MWh</b>
            </div>
          )}
        </div>
        <div>
          <h3>Chat (Curtailment Advisor)</h3>
          <ChatBox />
        </div>
      </div>
    </div>
  )
}
