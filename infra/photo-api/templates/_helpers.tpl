{{- define "photo-api.fullname" -}}
photo-api
{{- end }}

{{- define "photo-api.labels" -}}
app.kubernetes.io/name: {{ include "photo-api.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "photo-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "photo-api.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
