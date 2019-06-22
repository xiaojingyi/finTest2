from CCPRestSDK import REST

def sendTemplateSMS(to,datas,tempId):
    rest = REST(serverIP,serverPort,softVersion)
    rest.setAccount(accountSid,accountToken)
    rest.setAppId(appId)
    result = rest.sendTemplateSMS(to,datas,tempId)
    return result
