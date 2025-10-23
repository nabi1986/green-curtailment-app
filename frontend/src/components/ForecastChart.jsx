import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ResponsiveContainer } from 'recharts'

export default function ForecastChart({ data }) {
  const formatted = (data||[]).map(p => ({ 
    time: new Date(p.timestamp).getHours()+':00',
    gen: p.gen_mw,
    price: p.price_eur_mwh,
    risk: Math.round(p.curtailment_prob*100)
  }))
  return (
    <div style={{height:320, border:'1px solid #eee', borderRadius:8, padding:8}}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={formatted}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          {/* distinct colors */}
          <Line yAxisId="left"  type="monotone" dataKey="gen"   name="Gen (MW)"        stroke="#2563EB" dot={false}/>
          <Line yAxisId="right" type="monotone" dataKey="price" name="Price (€/MWh)"   stroke="#16A34A" dot={false}/>
          <Line yAxisId="right" type="monotone" dataKey="risk"  name="Curtailment (%)" stroke="#F59E0B" dot={false}/>
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
